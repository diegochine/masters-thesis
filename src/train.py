import os
import shutil

import hydra
import omegaconf
import torch
import wandb
from omegaconf import DictConfig
from torchrl.collectors import MultiSyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import SerialEnv, ParallelEnv
from tqdm import tqdm

from src.envs import VPPEnv
from src.utils.training import make_env, train_loop, get_agent_modules, seed_everything


def init_wandb(cfg):
    tags = [cfg.agent.algo]
    if cfg.tag:
        tags += [cfg.tag]
    if cfg.environment.params.variant != 'toy':
        tags += ['safety_layer'] if cfg.environment.params.safety_layer else []
        tags += [cfg.environment.params.variant]
    wandb_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(**cfg.wandb.setup, group=cfg.environment.params.variant, tags=tags, config=wandb_cfg,
               settings=wandb.Settings(start_method="thread"), reinit=True)
    # for net in nets:
    #     wandb.watch(net, **cfg.wandb.watch)
    wandb.define_metric("train/avg_score", summary="max", step_metric='train/iteration')
    wandb.define_metric("train/avg_violation", summary="min", step_metric='train/iteration')
    wandb.define_metric("train/avg_cost", summary="last", step_metric='train/iteration')
    wandb.define_metric("eval/avg_score", summary="max", step_metric='train/iteration')
    wandb.define_metric("eval/avg_violation", summary="min", step_metric='train/iteration')
    wandb.define_metric("eval/avg_cost", summary="last", step_metric='train/iteration')

    wandb.define_metric("final_eval/deterministic/all_scores", summary="last", step_metric='timestep_deterministic')
    wandb.define_metric("final_eval/deterministic/all_violations", summary="last", step_metric='timestep_deterministic')
    wandb.define_metric("final_eval/deterministic/all_costs", summary="last", step_metric='timestep_deterministic')
    wandb.define_metric("final_eval/deterministic/avg_storage_capacity", summary="last",
                        step_metric='timestep_deterministic')
    wandb.define_metric("final_eval/stochastic/all_scores", summary="last", step_metric='timestep_stochastic')
    wandb.define_metric("final_eval/stochastic/all_violations", summary="last", step_metric='timestep_stochastic')
    wandb.define_metric("final_eval/stochastic/all_costs", summary="last", step_metric='timestep_stochastic')
    wandb.define_metric("final_eval/stochastic/avg_storage_capacity", summary="last", step_metric='timestep_stochastic')

    wandb.define_metric("debug/actor_lr", step_metric='train/iteration')
    wandb.define_metric("debug/critic_lr", step_metric='train/iteration')
    wandb.define_metric("debug/lag_lr", step_metric='train/iteration')
    for a in VPPEnv.ACTIONS:
        wandb.define_metric(f"train/{a}", step_metric='train_episode')
        wandb.define_metric(f"eval/{a}", step_metric='train/iteration')
        wandb.define_metric(f"final_eval/deterministic/{a}", step_metric='timestep_deterministic')
        wandb.define_metric(f"final_eval/stochastic/{a}", step_metric='timestep_stochastic')

    wandb.define_metric("train/loss_lagrangian", step_metric='train_step')
    wandb.define_metric("train/loss_pi", step_metric='train_step')
    wandb.define_metric("train/loss_entropy", step_metric='train_step')
    wandb.define_metric("train/loss_r_critic", step_metric='train_step')
    wandb.define_metric("train/loss_c_critic", step_metric='train_step')
    wandb.define_metric("debug/lagrangian", step_metric='train/iteration')
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
    if not os.path.isdir(cfg.training.save_dir):
        os.makedirs(cfg.training.save_dir)

    # Set seed
    seed_everything(cfg.seed)

    # Set device and total frames
    device = torch.device(cfg.training.device)
    total_frames = cfg.training.frames_per_batch * cfg.training.iterations

    # Create env to initialize modules and normalization state dict
    env = make_env(device=device, instances=cfg.environment.instances.train,
                   **cfg.environment.params)
    loss_module, lag_module, policy_module, nets = get_agent_modules(env, cfg, device)
    t_state_dict = env.transform[0].state_dict()
    del env

    env_kwargs = [{'device': device, 't_state_dict': t_state_dict, 'wandb_run': None,
                   'instances': cfg.environment.instances.train, **cfg.environment.params}] * cfg.training.num_envs
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
    collector.set_seed(cfg.seed)
    replay_buffer = TensorDictReplayBuffer(
        batch_size=cfg.training.batch_size,
        storage=LazyMemmapStorage(cfg.training.frames_per_batch),
        prefetch=cfg.training.num_epochs,
        sampler=SamplerWithoutReplacement()  # for PPO only, ensures the entire dataset is used
    )

    valid_env = ParallelEnv(num_workers=len(cfg.environment.instances.valid), create_env_fn=make_env,
                            create_env_kwargs=[
                                {'device': device, 't_state_dict': t_state_dict, 'fixed_noise': True,
                                 **cfg.environment.params,
                                 'instances': [instance]} for instance in cfg.environment.instances.valid])
    if cfg.environment.instances.test not in (None, 'None'):  # wandb sweep sends None as string
        test_env = SerialEnv(num_workers=len(cfg.environment.instances.test), create_env_fn=make_env,
                             create_env_kwargs=[
                                 {'device': device, 't_state_dict': t_state_dict, 'fixed_noise': True,
                                  **cfg.environment.params,
                                  'instances': [instance]} for instance in cfg.environment.instances.test])
        test_env.reset()
    else:
        test_env = valid_env
    valid_env.reset()

    optim = torch.optim.Adam([
        {'params': [p for k, p in loss_module.named_parameters() if 'critic' in k], 'lr': cfg.agent.critic_lr},
        {'params': [p for k, p in loss_module.named_parameters() if 'actor' in k],
         'lr': cfg.agent.actor_lr, 'weight_decay': cfg.agent.actor_weight_decay},
    ], eps=1e-5)
    if cfg.agent.schedule:  # square-summable, non-summable step sizes
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.training.iterations, eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda _: 1.)

    pbar = tqdm(total=total_frames, desc="Training", unit=" frames")
    train_loop(cfg=cfg, collector=collector, device=device, valid_env=valid_env, test_env=test_env,
               loss_module=loss_module, lag_module=lag_module, optim=optim, pbar=pbar, policy_module=policy_module,
               replay_buffer=replay_buffer, scheduler=scheduler)

    # clean up
    shutil.rmtree(cfg.training.save_dir)
    wandb_dir = wandb.run.dir[:-6] if 'files' in wandb.run.dir else wandb.run.dir
    wandb.finish()
    print(f'Cleaning up {wandb_dir}')
    shutil.rmtree(wandb_dir)


if __name__ == "__main__":
    main()
