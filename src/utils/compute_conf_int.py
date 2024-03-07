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


def get_from_wandb(run, set, metrics, variant_key, store_keys):
    """Download run data from wandb.
    :param run: wandb.Run
    :param set: str; either 'test' or 'valid'
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

    metrics_dict = dict()
    for v1 in ('final', 'best'):
        for v2 in ('deterministic', 'stochastic'):
            keys = [f'{set}/{v1}/{v2}/{m}' for m in metrics]
            h = run.history(samples=1, keys=keys, pandas=False)
            del h[0]['_step']
            metrics_dict.update(h[0])
    return {**hyperparams,
            'cost_limit': run.config['agent']['lagrange']['params']['cost_limit'],
            **{k: v for k, v in metrics_dict.items()}}


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
    parser.add_argument('--hue', type=lambda x: None if x in ('null', 'None') else x, default='null')
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
             'agent.lagrange.type': 'lagrangian method', 'agent.lag_lr': 'naive-lag-lr',
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
    save_path = f"analysis/{'-'.join(args.sweep_id)}"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if not os.path.isfile(f'{save_path}/data.csv'):
        api = wandb.Api()

        runs = [r for s_id in args.sweep_id for r in api.sweep(f'unify/long-term-constraints/{s_id}').runs]

        map_f = partial(get_from_wandb, set=args.set, metrics=args.metrics,
                        variant_key=args.variant_key, store_keys=args.store_keys)
        with ThreadPoolExecutor(max_workers=25) as ex:
            print(f'Downloading data from wandb (Sweep ID: {args.sweep_id})')
            raw_data = ex.map(map_f, runs)
        df = pd.DataFrame(list(raw_data))
        df.to_csv(f'{save_path}/data.csv', index=False)
    else:
        print(f'Loading data from {save_path}/data.csv')
        df = pd.read_csv(f'{save_path}/data.csv')

    variants = df[args.variant_key].unique()
    df['cost limit'] = df['cost_limit']
    cost_limits = df['cost limit'].unique()
    var_palette = dict(zip(variants, sns.color_palette(n_colors=len(variants))))
    cl_palette = dict(zip(cost_limits, sns.color_palette(palette='Set2', n_colors=len(cost_limits))))
    if args.hue is not None:
        if args.hue == 'cost_limit':
            args.hue = 'cost limit'
            palette = 'Set2'
        else:
            palette = 'deep'
        hue_vals = df[args.hue].unique()
        assert len(hue_vals) > 1, f'There is only one unique val for {args.hue}, cannot use it as hue'
        palette = dict(zip(hue_vals, sns.color_palette(palette=palette, n_colors=len(hue_vals))))
        plot_kwargs = {'palette': palette, 'hue': args.hue, 'dodge': 0.5}
    else:
        plot_kwargs = {}

    metrics_to_label = {'avg_cost': 'cost',
                        'avg_score': 'return (normalized)',
                        'surrogate_score': '$abs(J_C^{\\theta} - d)$'}
    metrics_to_title = {'avg_cost': 'average cost',
                        'avg_score': 'average return',
                        'surrogate_score': 'constraint score'}

    for v in ('deterministic', 'stochastic'):
        for m in args.metrics:
            cplot = sns.catplot(df, x=args.variant_key, y=f'{args.set}/best/{v}/{m}', **plot_kwargs, kind='point',
                                errorbar=lambda x: compute_ci(x), linestyles='', capsize=.2, )
            # sns.swarmplot(df, x=args.variant_key, y=f'{args.set}/best/{v}/{m}', hue='cost_limit', palette=cl_palette,
            #               ax=cplot.ax, dodge=True, legend=False, size=5, edgecolor='black', linewidth=0.5, alpha=0.5)
            if m == 'avg_cost':
                for cl in cost_limits:
                    plt.axhline(cl, linewidth=1, linestyle='--', color=cl_palette[cl], label=cl)
                plt.ylim(0, 1100)
            elif m == 'avg_score':
                plt.ylim(0.92, 0.99)
            cplot.fig.subplots_adjust(top=.95)
            cplot.ax.set_title(f'95% CI for {metrics_to_title[m]} according to {sanitize(args.variant_key)}')
            cplot.ax.set_xlabel(sanitize(args.variant_key))
            cplot.ax.set_ylabel(metrics_to_label[m])
            cplot.fig.savefig(f'{save_path}/{sanitize(args.variant_key, m)}-{v[:3]}.png', dpi=300)
