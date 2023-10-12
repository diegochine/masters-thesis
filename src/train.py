import argparse

import hydra
import omegaconf
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torchrl.collectors import MultiSyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import SerialEnv
from tqdm import tqdm

from src.envs import VPPEnv
from src.utils import VARIANTS, RL_ALGOS, CONTROLLERS
from src.utils.training import make_env, train_loop, get_agent_modules, seed_everything


def get_args_dict():
    """Constructs CLI argument parser, and returns dict of arguments."""
    parser = argparse.ArgumentParser()

    # Main arguments
    parser.add_argument("logdir", type=str, help="Logging directory")
    parser.add_argument("-v", "--variant", type=str, choices=VARIANTS,
                        help="'toy': toy variant of the vpp problem (no battery);"
                             "'standard': standard variant of the vpp problem;"
                             "'cumulative': vpp problem with cumulative constraint on the battery")
    parser.add_argument("-a", "--algo", type=str, choices=RL_ALGOS, default='PPOLag',
                        help="Offline RL algorithms to use, 'PPOLag'")
    parser.add_argument("-c", "--controller", type=str, choices=CONTROLLERS,
                        help="Type of controller, 'rl' or 'unify'")

    # Additional configs
    parser.add_argument('-sl', '--safety-layer', action='store_true',
                        help="If True, use safety layer to correct unfeasible actions at training time."
                             "Safety Layer is always enabled at testing time to ensure action feasibility.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = vars(parser.parse_args())
    return args


def init_wandb(cfg):
    tags = [cfg.agent.algo]
    if cfg.environment.variant != 'toy':
        tags += ['safety_layer'] if cfg.environment.safety_layer else []
        tags += [cfg.environment.controller]
        tags += list(map(lambda i: str(i), OmegaConf.to_object(cfg.environment.instances)))
    wandb_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(**cfg.wandb.setup, group=cfg.environment.variant, tags=tags, config=wandb_cfg,
               settings=wandb.Settings(start_method="thread"))
    # for net in nets:
    #     wandb.watch(net, **cfg.wandb.watch)
    wandb.define_metric("train/avg_score", summary="max", step_metric='train/iteration')
    wandb.define_metric("train/avg_violation", summary="min", step_metric='train/iteration')
    wandb.define_metric("train/max_steps", summary="max", step_metric='train/iteration')
    wandb.define_metric("eval/avg_score", summary="max", step_metric='train/iteration')
    wandb.define_metric("eval/avg_violation", summary="min", step_metric='train/iteration')
    wandb.define_metric("eval/all_scores", summary="max", step_metric='train/iteration')
    wandb.define_metric("eval/all_violations", summary="max", step_metric='train/iteration')
    wandb.define_metric("debug/actor_lr", step_metric='train/iteration')
    wandb.define_metric("debug/critic_lr", step_metric='train/iteration')
    wandb.define_metric("debug/lag_lr", step_metric='train/iteration')
    for a in VPPEnv.ACTIONS:
        wandb.define_metric(f"train/{a}", step_metric='train_episode')
        wandb.define_metric(f"eval/{a}", step_metric='train/iteration')

    wandb.define_metric("train/loss_lagrangian", step_metric='train_step')
    wandb.define_metric("train/loss_pi", step_metric='train_step')
    wandb.define_metric("train/loss_entropy", step_metric='train_step')
    wandb.define_metric("train/loss_r_critic", step_metric='train_step')
    wandb.define_metric("train/loss_c_critic", step_metric='train_step')
    wandb.define_metric("debug/lagrangian", step_metric='train_step')
    wandb.define_metric("debug/entropy", step_metric='train_step')
    wandb.define_metric("debug/preds_reward", step_metric='train_step')
    wandb.define_metric("debug/targets_reward", step_metric='train_step')
    wandb.define_metric("debug/preds_constraint", step_metric='train_step')
    wandb.define_metric("debug/targets_constraint", step_metric='train_step')
    wandb.define_metric("debug/approx_kl", step_metric='train_step')


@hydra.main(version_base="1.1", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """
    Entry point for training.
    :param cfg: DictConfig; Hydra configuration object.
    """

    # Set seed
    seed_everything(cfg.seed)

    # Set device and total frames
    device = torch.device(cfg.training.device)
    total_frames = cfg.training.frames_per_batch * cfg.training.iterations

    # Create env to initialize modules and normalization state dict
    env = make_env(device=device, **cfg.environment)
    loss_module, policy_module, nets = get_agent_modules(env, cfg, device)
    t_state_dict = env.transform[0].state_dict()
    del env

    env_kwargs = [{'device': device, 't_state_dict': t_state_dict, 'wandb_run': None,
                   **cfg.environment}] * cfg.training.num_envs
    # Initialize wandb
    if cfg.wandb.use_wandb:
        init_wandb(cfg)
        env_kwargs[0]['wandb_run'] = wandb.run  # only log from the first training env

    # Create data collector and replay buffer
    collector = MultiSyncDataCollector(
        create_env_fn=[make_env] * cfg.training.num_envs,
        create_env_kwargs=env_kwargs,  # TODO pr torchrl to fix this
        policy=policy_module,
        frames_per_batch=cfg.training.frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
    )
    replay_buffer = TensorDictReplayBuffer(
        batch_size=cfg.training.batch_size,
        storage=LazyMemmapStorage(cfg.training.frames_per_batch),
        prefetch=cfg.training.num_epochs,
        sampler=SamplerWithoutReplacement()  # for PPO only, ensures the entire dataset is used
    )
    eval_env = SerialEnv(num_workers=len(cfg.environment.instances),
                         create_env_fn=make_env,
                         create_env_kwargs=[
                             {'device': device, 't_state_dict': t_state_dict, **cfg.environment,
                              'instances': [instance]} for instance in cfg.environment.instances])
    eval_env.reset()

    optim = torch.optim.Adam([
        {'params': [p for k, p in loss_module.named_parameters() if 'actor' in k],
         'lr': cfg.agent.actor_lr, 'weight_decay': cfg.agent.actor_weight_decay},
        {'params': [p for k, p in loss_module.named_parameters() if 'critic' in k], 'lr': cfg.agent.critic_lr},
        {'params': [p for k, p in loss_module.named_parameters() if 'lag' in k], 'lr': cfg.agent.lag_lr}
    ])
    if cfg.agent.schedule:
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optim,
                                                              lr_lambda=[lambda _: cfg.agent.schedule_factor,
                                                                         lambda _: cfg.agent.schedule_factor,
                                                                         lambda _: 1.])
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda _: 1.)

    pbar = tqdm(total=total_frames, desc="Training", unit=" frames")
    train_loop(cfg, collector, device, eval_env, loss_module, optim, pbar, policy_module, replay_buffer, scheduler)

    collector.shutdown()
    pbar.close()


if __name__ == "__main__":
    main()
