from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser
from functools import partial
import os

import wandb
from scipy.stats import bootstrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_from_wandb(run, set, use_final, use_deterministic, metrics, variant_key, store_keys):
    """Download run data from wandb.
    :param run: wandb.Run
    :param set: str; either 'test' or 'valid'
    :param use_final: bool; if True, consider the end-training policy, else consider the best policy. Defaults to False.
    :param use_deterministic: bool; if True, consider deterministic evaluation, otherwise the stochastic one.
        Defaults to False.
    :param metrics: tuple of str; the metrics to download.
    :param variant_key: str; the key to use to identify the variant;
        it must be a sequence of config keys separated by "." e.g. "environment.instances.train".
    :return: str, dict; the variant and the metrics used to compute intervals for this run.
    """

    def unwrap_config(k):
        cfg = run.config
        for k in k.split('.'):
            cfg = cfg[k]
        if isinstance(cfg, str):
            cfg = cfg.upper()
        elif isinstance(cfg, list):
            assert 'environment.instances.train' in {variant_key, *store_keys}, \
                f'"environment.instances.train" must be present if variant is a list, received: {variant_key}'
            cfg = f"{len(cfg)} instances"
        else:
            cfg = str(cfg)
        return cfg

    hyperparams = {k: unwrap_config(k) for k in [variant_key, *store_keys]}
    prefix = f'{set}/{"final" if use_final or run.config["agent"]["algo"] == "ppo" else "best"}/{"deterministic" if use_deterministic else "stochastic"}'
    keys = [f'{prefix}/{m}' for m in metrics]
    h = run.history(samples=1, keys=keys, pandas=False)

    return {**hyperparams,
            'cost_limit': run.config['agent']['lagrange']['params']['cost_limit'],
            **{k.split('/')[-1]: h[0][k] for k in keys}}


def get_args():
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument('sweep_id', nargs='+')
    parser.add_argument('--set', type=str, default='test')
    parser.add_argument('--use_final', action='store_true')
    parser.add_argument('--use_deterministic', action='store_true')
    parser.add_argument('--metrics', nargs='+',
                        default=('avg_cost', 'avg_score', 'surrogate_score'))
    parser.add_argument('--variant-key', type=str, default='agent.algo')
    parser.add_argument('--store-keys', nargs='+', default=tuple())
    return parser.parse_args()


def compute_ci(x):
    """Compute bootstrap confidence interval for a given array of values."""
    ci = bootstrap([x], statistic=np.mean, random_state=42).confidence_interval
    return ci.low, ci.high


def sanitize(variant_key, metric=None):
    mdict = {'avg_cost': 'cost', 'avg_score': 'score', 'surrogate_score': 'surr'}
    vdict = {'environment.params.variant': 'variant', 'environment.instances.train': 'train-instances',
             'training.batch_size': 'batch-size', 'training.num_epochs': 'epochs',
             'training.frames_per_batch': 'frames-per-batch', 'agent.algo': 'algo',
             'actor.net_spec.num_cells': 'actor-size', 'critic.net_spec.num_cells': 'critic-size',
             'agent.actor_lr': 'actor-lr', 'agent.critic_lr': 'critic-lr',
             'agent.lagrange.type': 'lagrange-algo', 'agent.lag_lr': 'naive-lag-lr',
             'agent.lagrange.params.initial_value': 'lagrangian-init',
             'agent.loss_module_lag.reward_scale': 'reward_scale', 'agent.loss_module_lag.cost_scale': 'cost_scale',
             'agent.lagrange.params.kd': 'pid-kd', 'agent.lagrange.params.ki': 'pid-ki', 'agent.lagrange.params.kp': 'pid-kp',
             'agent.use_beta': 'pid-beta'}

    if metric is not None:
        return f'{vdict[variant_key]}-{mdict[metric]}'
    else:
        return vdict[variant_key]


if __name__ == '__main__':
    args = get_args()
    print('Running with args:\n', args)
    save_path = f"cis/{'-'.join(args.sweep_id)}"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if not os.path.isfile(f'{save_path}/data.csv'):
        api = wandb.Api()

        runs = [r for s_id in args.sweep_id for r in api.sweep(f'unify/long-term-constraints/{s_id}').runs]

        map_f = partial(get_from_wandb, set=args.set, use_final=args.use_final,
                        use_deterministic=args.use_deterministic, metrics=args.metrics,
                        variant_key=args.variant_key, store_keys=args.store_keys)
        with ThreadPoolExecutor(max_workers=100) as ex:
            print(f'Downloading data from wandb (Sweep ID: {args.sweep_id})')
            raw_data = ex.map(map_f, runs)
        df = pd.DataFrame(list(raw_data))
        df.to_csv(f'{save_path}/data.csv', index=False)
    else:
        print(f'Loading data from {save_path}/data.csv')
        df = pd.read_csv(f'{save_path}/data.csv')

    variants = df[args.variant_key].unique()
    cost_limits = df['cost_limit'].unique()
    cl_palette = dict(zip(cost_limits, sns.color_palette(n_colors=len(cost_limits))))
    ys = {'avg_cost': 'Average cost', 'avg_score': 'Average return (normalized)',
          'surrogate_score': 'Surrogate score'}

    for m in args.metrics:
        cplot = sns.catplot(df, x=args.variant_key, y=m, hue='cost_limit', kind='bar', errorbar=lambda x: compute_ci(x),
                            palette=cl_palette, alpha=0.5)
        if m == 'avg_cost':
            for cl in cost_limits:
                plt.axhline(cl, linewidth=1, linestyle='--', color=cl_palette[cl], label=cl)
        elif m == 'avg_score':
            plt.ylim(0.85, 0.99)
        cplot.fig.subplots_adjust(top=.95)
        cplot.ax.set_title(f'{ys[m]} according to {sanitize(args.variant_key)}')
        cplot.ax.set_xlabel(sanitize(args.variant_key))
        cplot.ax.set_ylabel(ys[m])
        cplot.fig.savefig(f'{save_path}/{sanitize(args.variant_key, m)}.png')
