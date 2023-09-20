import os
from typing import Tuple

import hydra
import numpy as np
import pandas as pd

import torch
import tqdm
import wandb
from omegaconf import ListConfig, DictConfig
from tensordict import TensorDictBase
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torch import nn, Tensor
from torchrl.collectors import DataCollectorBase
from torchrl.data import TensorDictReplayBuffer, UnboundedDiscreteTensorSpec
from torchrl.envs import TransformedEnv, Compose, ObservationNorm, StepCounter, RewardSum, check_env_specs, \
    RewardScaling, default_info_dict_reader, EnvBase
from torchrl.envs.libs.gym import GymWrapper
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, IndependentNormal, TanhNormal, TruncatedNormal
from torchrl.objectives import LossModule
from torchrl.objectives.value import GAE
from torchrl.envs.utils import ExplorationType, set_exploration_type

from src.envs import StandardVPPEnv, CumulativeVPPEnv, SafeGridWorld
from src.agents import PPOLagLoss, NaiveLagrange, PIDLagrange

########################################################################################################################

TIMESTEP_IN_A_DAY = 96

VARIANTS = ['toy', 'standard', 'cumulative']
CONTROLLERS = ['rl', 'unify']
RL_ALGOS = ['ppolag']

wandb_running = lambda: wandb.run is not None


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


########################################################################################################################
def make_env(device: torch.device,
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
             reward_loc: float = 0.,
             reward_scale: float = 1.,
             t_state_dict: dict | None = None,
             **kwargs) -> TransformedEnv:
    """Environment factory function.
        :param device: torch.device to use
        :param variant: environment variant, one of {toy, standard, cumulative}
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
        :param reward_loc: reward location (mean)
        :param reward_scale: reward scale (std dev)
        :param t_state_dict: state dict of observation normalization; if not provided, initialize stats
    """
    if variant == 'toy':
        base_env = SafeGridWorld(grid_size=5)
    elif variant in {'standard', 'cumulative'}:
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

        if variant == 'standard':
            base_env = StandardVPPEnv(predictions=train_predictions,
                                      shift=shift,
                                      c_grid=c_grid,
                                      controller=controller,
                                      noise_std_dev=noise_std_dev,
                                      savepath=None,
                                      use_safety_layer=safety_layer,
                                      bound_storage_in=False,
                                      wandb_run=wandb_run)
        else:
            base_env = CumulativeVPPEnv(predictions=train_predictions,
                                        shift=shift,
                                        c_grid=c_grid,
                                        controller=controller,
                                        noise_std_dev=noise_std_dev,
                                        savepath=None,
                                        use_safety_layer=safety_layer,
                                        wandb_run=wandb_run)
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
    if reward_loc != 0. or reward_scale != 1.:
        transforms.append(RewardScaling(reward_loc, reward_scale))  # scale rewards

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

class ActorNet(nn.Module):

    def __init__(self, cfg: DictConfig, env: EnvBase, device: torch.device):
        super().__init__()
        self.state_dependent_std = cfg.agent.state_dependent_std
        if self.state_dependent_std:
            self.net = nn.Sequential(
                MLP(in_features=env.observation_spec['observation'].shape[0],
                    out_features=env.action_spec.shape[-1],
                    device=device,
                    activation_class=nn.ReLU,
                    **cfg.actor.net_spec),
                NormalParamExtractor(),
            )
        else:
            self.log_std = nn.Parameter(torch.full((env.action_spec.shape[-1],),
                                                   np.log(cfg.agent.std_dev_init),
                                                   device=device),
                                        requires_grad=cfg.agent.std_dev_trainable)
            self.net = nn.Sequential(
                MLP(in_features=env.observation_spec['observation'].shape[0],
                    out_features=env.action_spec.shape[-1],
                    device=device,
                    activation_class=nn.ReLU,
                    **cfg.actor.net_spec),
            )
        # initialize last layer of actor_net to produce actions close to zero at the beginning
        torch.nn.init.uniform_(self.net[0][-1].weight, -1e-3, 1e-3)

    def forward(self, x):
        x = self.net(x)
        if self.state_dependent_std:
            return x
        else:
            std = torch.exp(self.log_std)
            if len(x.shape) > 1:  # must add batch dim
                std = torch.tile(std, (x.shape[0], 1))
            return x, std


def get_agent_modules(env: EnvBase,
                      cfg: DictConfig,
                      device: torch.device) -> (LossModule, ProbabilisticActor, Tuple[nn.Module]):
    """Creates and initializes agent modules.
    :param env: environment used for initialization.
    :param cfg: Hydra config.
    :param device: torch device to use.
    """

    def init_module(module, name):
        print(f"Running {name}:", module(env.reset()))

    if cfg.agent.algo == 'ppolag':
        actor_net = ActorNet(cfg=cfg, env=env, device=device)
        distribution_class = IndependentNormal if cfg.agent.actor_dist_bound <= 0 else TruncatedNormal
        distribution_kwargs = None if cfg.agent.actor_dist_bound <= 0 else {'min': -cfg.agent.actor_dist_bound,
                                                                            'max': cfg.agent.actor_dist_bound}
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
        in_features = env.observation_spec['observation'].shape[0]
        critic_net = MLP(in_features=in_features, out_features=1, device=device, activation_class=nn.ReLU,
                         **cfg.critic.net_spec)
        r_value_module = ValueOperator(
            module=critic_net,
            in_keys=["observation"],
            out_keys=["r_state_value"],
        )
        safe_critic_net = MLP(in_features=in_features, out_features=1, device=device, activation_class=nn.ReLU,
                              # ensure cost critic only outputs positive values by adding final ReLU
                              activate_last_layer=cfg.critic.constraint_activation,
                              **cfg.critic.net_spec)
        c_value_module = ValueOperator(
            module=safe_critic_net,
            in_keys=["observation"],
            out_keys=["c_state_value"],
        )

        init_module(policy_module, 'policy')
        init_module(r_value_module, 'reward value function')
        init_module(c_value_module, 'constraint value function')

        r_advantage_module = GAE(
            value_network=r_value_module, average_gae=True,
            advantage_key="r_advantage", value_target_key="r_value_target", value_key='r_state_value',
            **cfg.agent.estimator,
        )
        c_advantage_module = GAE(
            value_network=c_value_module, average_gae=True,
            advantage_key="c_advantage", value_target_key="c_value_target", value_key='c_state_value',
            **cfg.agent.estimator,
        )
        if cfg.agent.lagrange.type == 'naive':
            lagrangian = NaiveLagrange(initial_value=cfg.agent.lagrange.params.initial_value,
                                       cost_limit=cfg.agent.lagrange.params.cost_limit)
        elif cfg.agent.lagrange.type == 'pid':
            lagrangian = PIDLagrange(**cfg.agent.lagrange.params)
        else:
            raise ValueError(f'Unknown lagrange type {cfg.agent.lagrange}, either naive or pid')
        loss_module = PPOLagLoss(
            actor=policy_module,
            critic=r_value_module,
            safe_critic=c_value_module,
            r_value_estimator=r_advantage_module,
            c_value_estimator=c_advantage_module,
            lagrangian=lagrangian,
            r_advantage_key=r_advantage_module.advantage_key,
            c_advantage_key=c_advantage_module.advantage_key,
            r_value_target_key=r_advantage_module.value_target_key,
            c_value_target_key=c_advantage_module.value_target_key,
            r_value_key=r_advantage_module.value_key,
            c_value_key=c_advantage_module.value_key,
            **cfg.agent.loss_module
        )
    else:
        raise ValueError(f'Unrecognized algo {cfg.agent.algo}')
    return loss_module, policy_module, (actor_net, critic_net, safe_critic_net)


def evaluate(eval_env: EnvBase, policy_module: ProbabilisticActor, optimal_scores: dict):
    """Evaluate the policy on the evaluation environment.
    :param eval_env: environment to evaluate on.
    :param policy_module: policy to evaluate.
    :param optimal_scores: optimal costs for each instance in the evaluation environment.
    """
    with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
        # execute a rollout with the trained policy
        eval_rollout = eval_env.rollout(100, policy_module)
        rewards = get_rollout_scores(eval_rollout, reduce=False)
        # normalize scores according to optimal score of each instance
        optimal_scores_tensor = torch.as_tensor([int(optimal_scores[int(instance)])
                                                 for instance in rewards[:, 2]])
        rewards[:, 0] = -optimal_scores_tensor / rewards[:, 0]
        rewards[:, 1] = rewards[:, 1] / 500  # FIXME should not be hardcoded, works for cap_max = 1000, c = 0.5
        eval_log = {'eval/avg_score': rewards[:, 0].mean().item(),
                    'eval/avg_violation': rewards[:, 1].mean().item(),
                    'eval/all_scores': wandb.Histogram(np_histogram=np.histogram(rewards[:, 0])),
                    'eval/all_violations': wandb.Histogram(np_histogram=np.histogram(rewards[:, 1]))
                    }
        eval_str = f"EVAL: avg cumreward = {eval_log['eval/avg_score']: 1.2f}, " \
                   f"avg violation = {eval_log['eval/avg_violation']: 1.2f}"
        histories = list(eval_env.history)
        actions_log = {f'eval/{k}': np.array([[v] for h in histories for v in h[k]]) for k in histories[0].keys()}
        eval_env.reset()  # reset the environment after the eval rollout
        del eval_rollout
    return {**eval_log, **actions_log}, eval_str


def get_rollout_scores(rollout_td: TensorDictBase, reduce: bool = True) -> Tensor | Tuple[float, float]:
    """Get the scores from a rollout.
    :param rollout_td: rollout to get the scores from.
    :param reduce: whether to reduce the scores to a single value (mean) or not.
    :return: (mean score, mean violation) if reduce;
        else, tensor of shape (n_dones, 3) with scores, violations and instance number.
    """
    dones = (rollout_td[('next', 'done')] | rollout_td[('next', 'truncated')]).reshape(-1)
    cumrewards = rollout_td['next', 'episode_reward'].reshape(-1, 2)
    instances = rollout_td['instance'].reshape(-1, 1)
    rewards = torch.cat([cumrewards, instances], dim=1)[dones.squeeze()]
    if reduce:
        avg_score = rewards[:, 0].mean().item()
        avg_violation = rewards[:, 1].mean().item()
        return avg_score, avg_violation
    else:
        return rewards


def train_loop(cfg: DictConfig,
               collector: DataCollectorBase,
               device: torch.device,
               eval_env: EnvBase,
               loss_module: LossModule,
               optim: torch.optim.Optimizer,
               pbar: tqdm.tqdm,
               policy_module: ProbabilisticActor,
               replay_buffer: TensorDictReplayBuffer,
               scheduler: torch.optim.lr_scheduler.LRScheduler) -> None:
    """Main training loop.
    :param cfg: hydra config
    :param collector: torchrl data collector
    :param device: torch device
    :param eval_env: evaluation environment
    :param loss_module: module used to compute the loss
    :param optim: optimizer
    :param pbar: tqdm progress bar
    :param policy_module: policy module object, used for evaluation
    :param replay_buffer: replay buffer
    :param scheduler: learning rate scheduler
    """
    optimal_scores = {instance: np.load(hydra.utils.to_absolute_path(f'src/data/oracle/{instance}_cost.npy'))
                      for instance in cfg.environment.instances}
    num_batches = cfg.training.frames_per_batch // cfg.training.batch_size
    train_step = 0
    # Iterate over the collector until it reaches frames_per_batch frames
    for it, rollout_td in enumerate(collector):
        # get dones to compute average cumulative reward and constraint violation
        rewards = get_rollout_scores(rollout_td, reduce=False)
        avg_violation = rewards[:, 1].mean().item()
        rollout_td['avg_violation'] = torch.full(rollout_td.batch_size, avg_violation)
        for epoch in range(cfg.training.num_epochs):
            data_view = rollout_td.reshape(-1)
            replay_buffer.extend(data_view.cpu())
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
        optimal_scores_tensor = torch.as_tensor([int(optimal_scores[int(instance)])
                                                 for instance in rewards[:, 2]])
        train_log = {'train/iteration': it,
                     'train/avg_score': (-optimal_scores_tensor / rewards[:, 0]).mean().item(),
                     'train/avg_violation': avg_violation / 500,
                     'train/max_steps': rollout_td["step_count"].max().item(),
                     'train/loc_cvirt_in': wandb.Histogram(np_histogram=np.histogram(rollout_td['loc'][:, 0])),
                     'train/scale_cvirt_in': wandb.Histogram(np_histogram=np.histogram(rollout_td['scale'][:, 0])),
                     'train/loc_cvirt_out': wandb.Histogram(np_histogram=np.histogram(rollout_td['loc'][:, 1])),
                     'train/scale_cvirt_out': wandb.Histogram(np_histogram=np.histogram(rollout_td['scale'][:, 1])),
                     'debug/actor_lr': optim.param_groups[0]["lr"],
                     'debug/critic_lr': optim.param_groups[1]["lr"],
                     'debug/lag_lr': optim.param_groups[2]["lr"]}

        pbar.update(rollout_td.numel())
        train_str = f"TRAIN: avg cumreward = {train_log['train/avg_score']: 1.2f}, " \
                    f"avg violation = {train_log['train/avg_violation']: 1.2f}, " \
                    f"max steps = {train_log['train/max_steps']: 2d}"
        eval_log, eval_str = evaluate(eval_env, policy_module, optimal_scores)

        pbar.set_description(f"{train_str} | {eval_str} ")
        if cfg.wandb.use_wandb:
            wandb.log({**train_log, **eval_log})
