import os

import numpy as np
import pandas as pd

import torch
import wandb
from omegaconf import ListConfig
from tensordict.nn import NormalParamExtractor, TensorDictModule, set_interaction_type, InteractionType
from torch import nn
from torchrl.envs import TransformedEnv, Compose, ObservationNorm, StepCounter, RewardSum, check_env_specs, \
    RewardScaling
from torchrl.envs.libs.gym import GymWrapper
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives.value import GAE
from torchrl.envs.utils import ExplorationType, set_exploration_type

from src.envs import StandardVPPEnv, CumulativeVPPEnv, SafeGridWorld
from src.agents import PPOLagLoss

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
             wandb_log: bool = True,
             max_steps: int = 100,
             iter_init_stats: int = 1000,
             reward_loc: float = 0.,
             reward_scale: float = 1.,
             t_state_dict: dict | None = None,
             **kwargs):
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
                                      wandb_log=wandb_log)
        elif variant == 'cumulative':
            base_env = CumulativeVPPEnv(predictions=train_predictions,
                                        shift=shift,
                                        c_grid=c_grid,
                                        controller=controller,
                                        noise_std_dev=noise_std_dev,
                                        savepath=None,
                                        use_safety_layer=safety_layer,
                                        wandb_log=wandb_log)
    else:
        raise ValueError(f'Variant name must be in {VARIANTS}')

    torchrl_env = GymWrapper(base_env, device=device)
    e = TransformedEnv(
        torchrl_env,
        Compose(
            ObservationNorm(in_keys=["observation"]),  # normalize observations
            StepCounter(max_steps=max_steps),  # maximum steps per episode
            RewardSum(),  # track sum of rewards over episodes
            RewardScaling(reward_loc, reward_scale),  # scale rewards
        ),
    )
    if t_state_dict is not None:
        e.transform[0].init_stats(num_iter=3, reduce_dim=0, cat_dim=0)
        e.transform[0].load_state_dict(t_state_dict)
    elif iter_init_stats > 0:
        e.transform[0].init_stats(num_iter=iter_init_stats, reduce_dim=0, cat_dim=0)
    # print("normalization constant shape:", e.transform[0].loc.shape)
    # print("observation_spec:", e.observation_spec)
    # print("reward_spec:", e.reward_spec)
    # print("done_spec:", e.done_spec)
    # print("action_spec:", e.action_spec)
    # print("state_spec:", environment.state_spec)
    check_env_specs(e)
    return e


########################################################################################################################

def get_agent_modules(env, cfg, device):
    def init_module(module, name):
        print(f"Running {name}:", module(env.reset()))

    if cfg.agent.algo == 'ppolag':
        in_features = env.observation_spec['observation'].shape[0]
        actor_net = nn.Sequential(
            MLP(in_features=in_features,
                out_features=2 * env.action_spec.shape[-1],
                device=device,
                activation_class=nn.ReLU,
                **cfg.actor.net_spec),
            NormalParamExtractor(),
        )
        # initialize last layer of actor_net to produce actions close to zero at the beginning
        torch.nn.init.uniform_(actor_net[0][-1].weight, -1e-3, 1e-3)
        policy_module = ProbabilisticActor(
            module=TensorDictModule(
                module=actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
            ),
            spec=env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": cfg.actor.distribution_min,
                "max": cfg.actor.distribution_max,
            },
            return_log_prob=True,  # we'll need the log-prob for the numerator of the importance weights
            default_interaction_type=ExplorationType.RANDOM,
        )
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
        loss_module = PPOLagLoss(
            actor=policy_module,
            critic=r_value_module,
            safe_critic=c_value_module,
            r_value_estimator=r_advantage_module,
            c_value_estimator=c_advantage_module,
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


def train_loop(cfg, collector, device, eval_env, logs, loss_module, optim, pbar, policy_module, replay_buffer,
               scheduler):
    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for i, tensordict_data in enumerate(collector):
        # we now have a batch of data to work with. Let's learn something from it.
        dones = (tensordict_data[('next', 'done')] | tensordict_data[('next', 'truncated')])
        rewards = tensordict_data['next', 'episode_reward'][dones.squeeze()]
        avg_score = rewards[:, 0].mean().item()
        # We need the average violation to update the lagrangian
        avg_violation = rewards[:, 1].mean().item()
        tensordict_data['avg_violation'] = torch.full(tensordict_data.batch_size, avg_violation)
        for epoch in range(cfg.training.num_epochs):
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for b in range(cfg.training.frames_per_batch // cfg.training.batch_size):
                subdata = replay_buffer.sample(cfg.training.batch_size)
                loss_info = loss_module(subdata.to(device),
                                        train_lagrangian=(epoch + b == 0))  # update lagrangian only on the first step
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
                    wandb.log(batch_log)

        train_log = {'train/iteration': i,
                     'train/avg_score': avg_score,
                     'train/avg_violation': avg_violation,
                     'train/max_steps': tensordict_data["step_count"].max().item(),
                     'debug/actor_lr': optim.param_groups[0]["lr"],
                     'debug/critic_lr': optim.param_groups[1]["lr"],
                     'debug/lag_lr': optim.param_groups[2]["lr"]}

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        train_str = f"TRAIN: avg cumreward = {train_log['train/avg_score']: 4.0f}, " \
                    f"avg violation = {train_log['train/avg_violation']: 4.0f}, " \
                    f"max steps = {train_log['train/max_steps']: 2d}"
        with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = eval_env.rollout(1000, policy_module)
            dones = (eval_rollout[('next', 'done')] | eval_rollout[('next', 'truncated')])
            rewards = eval_rollout['next', 'episode_reward'][dones.squeeze()]
            avg_score = rewards[:, 0].mean().item()
            avg_violation = rewards[:, 1].mean().item()
            eval_log = {'eval/avg_score': avg_score,
                        'eval/avg_violation': avg_violation,
                        'eval/avg_sv': avg_score - avg_violation,  # objective of wandb sweeps (maximize)
                        }
            eval_str = f"EVAL: avg cumreward = {eval_log['eval/avg_score']: 4.0f}, " \
                       f"avg violation = {eval_log['eval/avg_violation']: 4.0f}"
            eval_env.reset()  # reset the environment after the eval rollout
            del eval_rollout

        pbar.set_description(f"{train_str} | {eval_str} ")
        if cfg.wandb.use_wandb:
            wandb.log({**train_log, **eval_log})

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()
