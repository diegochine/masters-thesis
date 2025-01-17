import os
from typing import Tuple

import hydra
import numpy as np
import pandas as pd

import torch
import tqdm
import wandb
from omegaconf import ListConfig, DictConfig
from tensordict import TensorDictBase, TensorDict
from tensordict.nn import NormalParamExtractor, TensorDictModule, AddStateIndependentNormalScale
from torch import nn, Tensor
from torchrl.collectors import DataCollectorBase
from torchrl.data import TensorDictReplayBuffer, UnboundedDiscreteTensorSpec
from torchrl.envs import TransformedEnv, Compose, ObservationNorm, StepCounter, RewardSum, check_env_specs, \
    default_info_dict_reader, EnvBase
from torchrl.envs.libs.gym import GymWrapper
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, IndependentNormal, TruncatedNormal
from torchrl.objectives import LossModule, ClipPPOLoss, ValueEstimators
from torchrl.objectives.value import GAE
from torchrl.envs.utils import ExplorationType, set_exploration_type

from src.envs import StandardVPPEnv, CumulativeVPPEnv, SafeGridWorld
from src.algos import PPOLagLoss, NaiveLagrange, PIDLagrange, LagrangeBase

########################################################################################################################

TIMESTEP_IN_A_DAY = 96

VARIANTS = ['toy', 'standard', 'cumulative']
CONTROLLERS = ['rl', 'unify']
RL_ALGOS = ['ppo', 'ppolag']

wandb_running = lambda: wandb.run is not None


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


########################################################################################################################
def make_env(device: torch.device,
             env_type: str,
             variant: str,
             controller: str | None = None,
             predictions_path: str | None = None,
             prices_path: str | None = None,
             shifts_path: str | None = None,
             instances: float | list | None = None,
             noise_std_dev: float | int = 0.01,
             safety_layer: bool = False,
             wandb_run: wandb.sdk.wandb_run.Run | None = None,
             max_steps: int | None = None,
             iter_init_stats: int = 1000,
             t_state_dict: dict | None = None,
             cost_fn_type: str = 'sparse',
             storage_io_bound: int = 200,
             **kwargs) -> TransformedEnv:
    """Environment factory function.
        :param device: torch.device to use
        :param env_type: environment type, one of {toy, standard, cumulative}
        :param variant: environment variant, one of {cvirt_in, cvirt_out, both_cvirts}
        :param controller: controller type, one of {rl, unify}
        :param predictions_path: path to predictions file (only for standard and cumulative variants)
        :param prices_path: path to prices file (only for standard and cumulative variants)
        :param shifts_path: path to shifts file (only for standard and cumulative variants)
        :param instances: if float, percentage of test instances; if list, list of test instances;
                if None, randomly choose an instance
        :param noise_std_dev: standard deviation of noise
        :param safety_layer: whether to use safety layer
        :param wandb_run: optional, wandb.Run object used to log
        :param max_steps: maximum number of steps per episode
        :param iter_init_stats: number of iterations to initialize stats
        :param t_state_dict: state dict of observation normalization; if not provided, initialize stats
    """
    if env_type == 'toy':
        base_env = SafeGridWorld(grid_size=5)
    elif env_type in {'standard', 'cumulative'}:
        assert all(p is not None for p in [controller, predictions_path, prices_path, shifts_path])
        assert os.path.isfile(predictions_path), f"{predictions_path} does not exist"
        assert os.path.isfile(prices_path), f"{prices_path} does not exist"
        assert os.path.isfile(shifts_path), f"{shifts_path} does not exist"
        predictions = pd.read_csv(predictions_path)
        shift = np.load(shifts_path)
        c_grid = np.load(prices_path)

        # Split between training and test
        if isinstance(instances, float):
            split_index = int(len(predictions) * (1 - instances))
            train_predictions = predictions[:split_index]
        elif isinstance(instances, (list, ListConfig)):
            train_predictions = predictions.iloc[instances]
        elif instances is None:  # randomly choose one instance
            indexes = np.arange(10000, dtype=np.int32)
            train_predictions = predictions.iloc[list(np.random.choice(indexes, size=1))]
        else:
            raise Exception("test_split must be list of int, float, or None")

        if env_type == 'standard':
            base_env = StandardVPPEnv(predictions=train_predictions,
                                      shift=shift,
                                      c_grid=c_grid,
                                      controller=controller,
                                      noise_std_dev=noise_std_dev,
                                      savepath=None,
                                      use_safety_layer=safety_layer,
                                      wandb_run=wandb_run,
                                      variant=variant,
                                      storage_io_bound=storage_io_bound,
                                      **kwargs)
        else:
            base_env = CumulativeVPPEnv(predictions=train_predictions,
                                        shift=shift,
                                        c_grid=c_grid,
                                        controller=controller,
                                        noise_std_dev=noise_std_dev,
                                        savepath=None,
                                        use_safety_layer=safety_layer,
                                        wandb_run=wandb_run,
                                        variant=variant,
                                        cost_fn_type=cost_fn_type,
                                        storage_io_bound=storage_io_bound,
                                        **kwargs)
    else:
        raise ValueError(f'Variant name must be in {VARIANTS}')

    # Make torchrl env with transforms
    info_reader = default_info_dict_reader(["instance"],
                                           spec=[UnboundedDiscreteTensorSpec(shape=torch.Size([1]), dtype=torch.int64)])
    torchrl_env = GymWrapper(base_env, device=device).set_info_dict_reader(info_reader)
    transforms = [
        ObservationNorm(in_keys=["observation"]),  # normalize observations
        StepCounter(max_steps=max_steps),  # maximum steps per episode
        RewardSum(),  # track sum of rewards over episodes
    ]

    e = TransformedEnv(torchrl_env, Compose(*transforms))

    # Initialize normalizations stats
    if t_state_dict is not None:
        e.transform[0].init_stats(num_iter=3, reduce_dim=0, cat_dim=0)
        e.transform[0].load_state_dict(t_state_dict)
    elif iter_init_stats > 0:
        print('Initializing stats...')
        e.transform[0].init_stats(num_iter=iter_init_stats, reduce_dim=0, cat_dim=0)
    check_env_specs(e)
    return e


########################################################################################################################

def orthogonal_init(mlp, scale):
    for layer in mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, scale)
            layer.bias.data.zero_()


def get_agent_modules(env: EnvBase,
                      cfg: DictConfig,
                      device: torch.device) -> (LossModule, LagrangeBase, ProbabilisticActor, Tuple[nn.Module]):
    """Creates and initializes agent modules.
    :param env: environment used for initialization.
    :param cfg: Hydra config.
    :param device: torch device to use.
    """

    def init_module(module, name):
        print(f"Running {name}:", module(env.reset()))

    in_features = env.observation_spec['observation'].shape[0]
    activation_class = nn.ReLU if cfg.agent.activation == 'relu' else nn.Tanh
    if cfg.environment.params.variant == 'flows-qp':
        distribution_class = TruncatedNormal
        distribution_kwargs = {'min': 0.0, 'max': 1.0}
    else:
        distribution_class = IndependentNormal if cfg.agent.actor_dist_bound <= 0 else TruncatedNormal
        distribution_kwargs = None if cfg.agent.actor_dist_bound <= 0 else {'min': -cfg.agent.actor_dist_bound,
                                                                            'max': cfg.agent.actor_dist_bound}
    if cfg.agent.state_dependent_std:  # scale outputted by net
        actor_net = nn.Sequential(
            MLP(in_features=in_features,
                out_features=2 * env.action_spec.shape[-1],
                device=device,
                activation_class=activation_class,
                **cfg.actor.net_spec),
            NormalParamExtractor(),
        )
    else:  # scale independent of state
        actor_net = nn.Sequential(
            MLP(in_features=in_features,
                out_features=env.action_spec.shape[-1],
                device=device,
                activation_class=activation_class,
                **cfg.actor.net_spec),
            AddStateIndependentNormalScale(env.action_spec.shape[-1])
        )

    if cfg.agent.orthogonal_init:
        orthogonal_init(actor_net[0], 1.0)
        torch.nn.init.orthogonal_(actor_net[0][-1].weight, 0.01)

    policy_module = ProbabilisticActor(
        module=TensorDictModule(
            module=actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
        ),
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,  # we'll need the log-prob for the numerator of the importance weights
        default_interaction_type=ExplorationType.RANDOM,
    )
    init_module(policy_module, 'policy')

    critic_net = MLP(in_features=in_features, out_features=1, device=device, activation_class=activation_class,
                     **cfg.critic.net_spec)
    if cfg.agent.orthogonal_init:
        orthogonal_init(critic_net, 0.01)

    if cfg.agent.algo == 'ppo':
        r_value_module = ValueOperator(module=critic_net, in_keys=["observation"], out_keys=["state_value"])
        init_module(r_value_module, 'reward value function')
        lag_module = None
        loss_module = ClipPPOLoss(actor=policy_module, critic=r_value_module, **cfg.agent.loss_module)
        loss_module.make_value_estimator(ValueEstimators.GAE, **cfg.agent.estimator)
        nets = (actor_net, critic_net)

    elif cfg.agent.algo == 'ppolag':
        safe_critic_net = MLP(in_features=in_features, out_features=1, device=device, activation_class=activation_class,
                              # ensure cost critic only outputs positive values by adding final ReLU
                              activate_last_layer=cfg.critic.constraint_activation,
                              **cfg.critic.net_spec)
        if cfg.agent.orthogonal_init:
            orthogonal_init(safe_critic_net, 0.01)

        r_value_module = ValueOperator(module=critic_net, in_keys=["observation"], out_keys=["r_state_value"])
        c_value_module = ValueOperator(module=safe_critic_net, in_keys=["observation"], out_keys=["c_state_value"])
        init_module(r_value_module, 'reward value function')
        init_module(c_value_module, 'constraint value function')

        r_advantage_module = GAE(value_network=r_value_module, average_gae=True, **cfg.agent.estimator)
        r_advantage_module.set_keys(advantage='r_advantage', value_target='r_value_target', value='r_state_value')
        c_advantage_module = GAE(value_network=c_value_module, average_gae=True, **cfg.agent.estimator)
        c_advantage_module.set_keys(advantage='c_advantage', value_target='c_value_target', value='c_state_value')

        if cfg.agent.lagrange.type == 'naive':
            lag_module = NaiveLagrange(initial_value=cfg.agent.lagrange.params.initial_value,
                                       cost_limit=cfg.agent.lagrange.params.cost_limit,
                                       lr=cfg.agent.lag_lr)
        elif cfg.agent.lagrange.type == 'pid':
            lag_module = PIDLagrange(**cfg.agent.lagrange.params)
        else:
            raise ValueError(f'Unknown lagrange type {cfg.agent.lagrange.type}, either naive or pid')

        loss_module = PPOLagLoss(
            actor=policy_module,
            critic=r_value_module,
            safe_critic=c_value_module,
            r_value_estimator=r_advantage_module,
            c_value_estimator=c_advantage_module,
            **cfg.agent.loss_module,
            **cfg.agent.loss_module_lag
        )
        nets = (actor_net, critic_net, safe_critic_net)
    else:
        raise ValueError(f'Unknown algorithm {cfg.agent.algo}, either ppo or ppolag')

    return loss_module, lag_module, policy_module, nets


def evaluate(eval_env: EnvBase, policy_module: ProbabilisticActor, optimal_scores: dict, cost_limit: int,
             log_type='avg', stoch_rollouts=10):
    """Evaluate the policy on the evaluation environment.
    :param eval_env: environment to evaluate on.
    :param policy_module: policy to evaluate.
    :param optimal_scores: optimal costs for each instance in the evaluation environment.
    :param cost_limit: cost limit of the agent.
    :param log_type: either 'avg' or 'all', whether to log all scores or just the average.
    """
    eval_log = dict()
    for explore_type, prefix in ((ExplorationType.MEAN, 'eval/deterministic'),
                                 (ExplorationType.RANDOM, 'eval/stochastic')):
        with set_exploration_type(explore_type), torch.no_grad():
            # execute a rollout (or n rollouts for stochastic) with the trained policy
            if 'stoch' in prefix:
                eval_rollout = torch.cat([eval_env.rollout(100, policy_module) for _ in range(stoch_rollouts)])
            else:
                eval_rollout = eval_env.rollout(100, policy_module)
            rewards = get_rollout_scores(eval_rollout)
            # normalize scores according to optimal score of each instance
            optimal_scores_tensor = torch.as_tensor([float(optimal_scores[int(instance)])
                                                     for instance in rewards[:, 2]])
            # note the following 3 vars refer to episode (cumulative) values
            all_costs = rewards[:, 1]
            all_scores = -optimal_scores_tensor / rewards[:, 0]
            all_violations = torch.maximum(torch.zeros_like(all_costs), all_costs - cost_limit) / (1000 - cost_limit)
            eval_log = {
                f'{prefix}/avg_score': all_scores.mean().item(),
                f'{prefix}/avg_cost': all_costs.mean().item(),
                f'{prefix}/avg_violation': all_violations.mean().item(),
                f'{prefix}/surrogate_score': abs(all_costs.mean().item() - cost_limit),
                **eval_log
            }

            histories = list(eval_env.history)
            if log_type == 'avg':  # logging action distribution across many episodes/envs, loses temporal info
                actions_log = {f'{prefix}/{k}': np.array([[v] for h in histories for v in h[k]]) for k in
                               histories[0].keys()}
            else:  # final evaluation, retain temporal info
                actions_log = {f'{prefix}/{k}': [v for v in histories[0][k]] for k in histories[0].keys()}

            eval_log = {**eval_log, **actions_log}
            eval_env.reset()  # reset the environment after the eval rollout
            del eval_rollout
    if log_type == 'avg':
        eval_str = f"[E] reward: {eval_log['eval/deterministic/avg_score']: 1.2f}, " \
                   f"violation: {eval_log['eval/deterministic/avg_violation']: 1.2f}, " \
                   f"cost: {eval_log['eval/deterministic/avg_cost']: 4.0f}"
    else:
        eval_str = ''

    return eval_log, eval_str


def get_rollout_scores(rollout_td: TensorDictBase) -> Tensor | Tuple[float, float]:
    """Get the scores from a rollout.
    :param rollout_td: rollout to get the scores from.
    :return: tensor of shape (n_dones, 3) with scores, violations and instance number.
    """
    dones = (rollout_td[('next', 'done')] | rollout_td[('next', 'truncated')]).reshape(-1)
    cumrewards = rollout_td['next', 'episode_reward'].reshape(-1, 2)
    instances = rollout_td['instance'].reshape(-1, 1)
    rewards = torch.cat([cumrewards, instances], dim=1)[dones.squeeze()]
    assert dones.sum().item() > 0, "No done found in rollout_td, increase frames_per_batch or decrease num_envs"
    return rewards


def final_evaluation(cost_limit, eval_env, optimal_scores, policy_module, prefix):
    eval_log, _ = evaluate(eval_env, policy_module, optimal_scores, cost_limit, log_type='all')
    assert all(isinstance(h, float) or len(h) == 96 for h in eval_log.values()), "Not all histories have length 96"
    for eval_type in ('deterministic', 'stochastic'):
        avg_storage = float(np.mean(eval_log[f'eval/{eval_type}/storage_capacity']))
        for timestep in range(96):
            log = {f'{prefix}/{k[5:]}': v[timestep]
                   for k, v in eval_log.items() if isinstance(v, list) and eval_type in k}
            log[f'{prefix}/{eval_type}/avg_storage_capacity'] = avg_storage
            wandb.log({**log, f'timestep_{eval_type}': timestep})

        wandb.log({f'{prefix}/{k[5:]}': v for k, v in eval_log.items() if isinstance(v, float) and eval_type in k})


def train_loop(cfg: DictConfig,
               collector: DataCollectorBase,
               device: torch.device,
               valid_env: EnvBase,
               test_env: EnvBase,
               loss_module: LossModule,
               lag_module: LagrangeBase,
               optim: torch.optim.Optimizer,
               pbar: tqdm.tqdm,
               policy_module: ProbabilisticActor,
               replay_buffer: TensorDictReplayBuffer,
               scheduler: torch.optim.lr_scheduler.LRScheduler) -> None:
    """Main training loop.
    :param cfg: hydra config
    :param collector: torchrl data collector
    :param device: torch device
    :param valid_env: validation environment
    :param test_env: test environment
    :param loss_module: module used to compute the loss
    :param optim: optimizer
    :param pbar: tqdm progress bar
    :param policy_module: policy module object, used for evaluation
    :param replay_buffer: replay buffer
    :param scheduler: learning rate scheduler
    """
    optimal_train_scores = {instance: np.load(hydra.utils.to_absolute_path(f'src/data/oracle/{instance}_cost.npy'))
                            for instance in cfg.environment.instances.train}
    optimal_valid_scores = {instance: np.load(hydra.utils.to_absolute_path(f'src/data/oracle/{instance}_cost.npy'))
                            for instance in cfg.environment.instances.valid}
    num_batches = cfg.training.frames_per_batch // cfg.training.batch_size
    cost_limit = cfg.agent.lagrange.params.cost_limit
    train_step = 0
    # keep track of best iterate
    best_score = 1000
    best_it = None

    # Iterate over the collector until it reaches frames_per_batch frames
    for it, rollout_td in enumerate(collector):
        # get dones to compute average cumulative reward and constraint cost and violation
        rewards = get_rollout_scores(rollout_td)
        avg_cost = rewards[:, 1].mean().item()
        avg_violation = (rewards[:, 1] - cost_limit).mean().item()
        surrogate_score = abs(avg_cost - cost_limit)
        if cfg.agent.lagrange.positive_violation:
            avg_violation = max(0., avg_violation)
        cost_td = TensorDict({'avg_cost': avg_cost, 'avg_violation': avg_violation}, [])

        if cfg.agent.algo == 'ppolag':
            # Optimization: update lagrangian
            loss_lagrangian = lag_module(cost_td, cost_scale=cfg.agent.loss_module_lag.cost_scale)
            loss_module.set_lagrangian(lag_module.get())

            if cfg.agent.use_beta:
                loss_module.update_beta(rollout_td, optim, cfg.training.max_grad_norm)
        else:
            rollout_td['next', 'reward'] = rollout_td['next', 'reward'][:, 0]
            loss_lagrangian = 0.
        replay_buffer.extend(rollout_td.reshape(-1).cpu())

        # Optimization: compute loss and optimize
        for epoch in range(cfg.training.num_epochs):
            for b in range(num_batches):
                subdata = replay_buffer.sample(cfg.training.batch_size)
                loss_info = loss_module(subdata.to(device))
                loss_value = sum(loss_info[k] for k in loss_info.keys() if k.startswith('loss_'))
                # Optimization: backward, grad clipping and optim step
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.training.max_grad_norm)
                optim.step()
                optim.zero_grad()
                # Log the losses and debug info
                if cfg.wandb.use_wandb:
                    batch_log = {f"{'train' if k.startswith('loss_') else 'debug'}/{k}":
                                     (v.item() if v.numel() == 1 else v.numpy())  # to log both scalars and arrays
                                 for k, v in loss_info.items()}
                    wandb.log({'train_step': train_step, **batch_log})
                    train_step += 1

        scheduler.step()

        train_log = {'train/iteration': it,
                     # normalize score according to each rollout instance's optimal score
                     'train/avg_score': (-torch.as_tensor([float(optimal_train_scores[int(instance)])
                                                           for instance in rewards[:, 2]])
                                         / rewards[:, 0]).mean().item(),
                     'train/avg_cost': avg_cost,
                     'train/avg_violation': max(0, avg_violation) / (1000 - cost_limit),
                     'train/surrogate_score': surrogate_score,
                     'train/loss_lagrangian': loss_lagrangian,
                     'debug/actor_lr': optim.param_groups[0]["lr"],
                     'debug/critic_lr': optim.param_groups[1]["lr"],
                     }
        if lag_module is not None:
            train_log = {**train_log, **{f'debug/{k}': v.item() for k, v in lag_module.get_logs().items()}}
        if cfg.agent.algo == 'ppolag' and cfg.agent.use_beta:
            train_log['debug/beta'] = loss_module.beta.item()
        if cfg.environment.params.variant == 'cvirt_in':
            train_log['train/loc_cvirt_in'] = wandb.Histogram(np_histogram=np.histogram(rollout_td['loc'][:, 0]))
            train_log['train/scale_cvirt_in'] = wandb.Histogram(np_histogram=np.histogram(rollout_td['scale'][:, 0]))
            train_log['train/loc_cvirt_out'] = 0.
            train_log['train/scale_cvirt_out'] = 0.
            train_log['train/loc_in_range'] = (rollout_td['loc'][:, 0].max() - rollout_td['loc'][:, 0].min()).item()
        elif cfg.environment.params.variant == 'cvirt_out':
            train_log['train/loc_cvirt_in'] = 0.
            train_log['train/scale_cvirt_in'] = 0.
            train_log['train/loc_cvirt_out'] = wandb.Histogram(np_histogram=np.histogram(rollout_td['loc'][:, 0]))
            train_log['train/scale_cvirt_out'] = wandb.Histogram(np_histogram=np.histogram(rollout_td['scale'][:, 0]))
            train_log['train/loc_out_range'] = (rollout_td['loc'][:, 0].max() - rollout_td['loc'][:, 0].min()).item()
        else:  # cfg.environment.params.variant == 'both_cvirts'
            train_log['train/loc_cvirt_in'] = wandb.Histogram(np_histogram=np.histogram(rollout_td['loc'][:, 0]))
            train_log['train/scale_cvirt_in'] = wandb.Histogram(np_histogram=np.histogram(rollout_td['scale'][:, 0]))
            train_log['train/loc_cvirt_out'] = wandb.Histogram(np_histogram=np.histogram(rollout_td['loc'][:, 1]))
            train_log['train/scale_cvirt_out'] = wandb.Histogram(np_histogram=np.histogram(rollout_td['scale'][:, 1]))
            train_log['train/loc_in_range'] = (rollout_td['loc'][:, 0].max() - rollout_td['loc'][:, 0].min()).item()
            train_log['train/loc_out_range'] = (rollout_td['loc'][:, 1].max() - rollout_td['loc'][:, 1].min()).item()

        pbar.update(rollout_td.numel())
        train_str = f"[T] reward: {train_log['train/avg_score']: 1.2f}, " \
                    f"violation: {train_log['train/avg_violation']: 1.2f}, " \
                    f"cost: {train_log['train/avg_cost']: 4.0f}, " \
                    f"lag: {lag_module.get() if lag_module is not None else 0.: 3.0f} "
        eval_log, eval_str = evaluate(valid_env, policy_module, optimal_valid_scores, cost_limit)

        if eval_log['eval/stochastic/surrogate_score'] < best_score:  # the closer to 0, the better
            best_score = eval_log['eval/stochastic/surrogate_score']
            best_it = it
            torch.save(policy_module.state_dict(), f'{cfg.training.save_dir}/policy_it{it}.pt')

        pbar.set_description(f"{train_str} | {eval_str} |")
        if cfg.wandb.use_wandb:
            wandb.log({**train_log, **eval_log})

    collector.shutdown()
    pbar.close()

    if cfg.wandb.use_wandb:  # final evaluation
        if cfg.environment.instances.test not in (None, 'None'):
            optimal_test_scores = {
                instance: np.load(hydra.utils.to_absolute_path(f'src/data/oracle/{instance}_cost.npy'))
                for instance in cfg.environment.instances.test}
            prefix = 'test'
        else:
            optimal_test_scores = optimal_valid_scores
            prefix = 'valid'
        print(f'Performing {prefix} with final model')
        torch.save(policy_module.state_dict(), os.path.join(wandb.run.dir, 'policy_final.pt'))
        final_evaluation(cost_limit, test_env, optimal_test_scores, policy_module, f'{prefix}/final')
        policy_module.load_state_dict(torch.load(f'{cfg.training.save_dir}/policy_it{best_it}.pt'))
        torch.save(policy_module.state_dict(), os.path.join(wandb.run.dir, 'policy_best.pt'))
        print(f'Perfoming {prefix} with best model')
        final_evaluation(cost_limit, test_env, optimal_test_scores, policy_module, f'{prefix}/best')
