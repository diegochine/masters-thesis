from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser
from functools import partial

import wandb
from scipy.stats import bootstrap
import numpy as np


def get_from_wandb(run, set, use_final, use_deterministic, metrics):
    """Download run data from wandb.
    :param run: wandb.Run
    :param set: str; either 'test' or 'valid'
    :param use_final: bool; if True, consider the end-training policy, else consider the best policy. Defaults to False.
    :param use_deterministic: bool; if True, consider deterministic evaluation, otherwise the stochastic one. Defaults to False.
    :param metrics: tuple of str; the metrics to download.
    :return: str, str, dict; the variant, the stratum and the metrics used to compute intervals for this run.
    """
    variant = run.config['agent']['algo']
    stratum = (run.config['environment']['instances']['train'], run.config['agent']['lagrange']['params']['cost_limit'])
    prefix = f'{set}/{"final" if use_final else "best"}/{"deterministic" if use_deterministic else "stochastic"}'
    keys = [f'{prefix}/{m}' for m in metrics]
    h = run.history(samples=1, keys=keys, pandas=False)
    return variant, stratum, {k.split('/')[-1]: h[0][k] for k in keys}


def get_args():
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument('sweep_id', type=str)
    parser.add_argument('--set', type=str, default='test')
    parser.add_argument('--use_final', action='store_true')
    parser.add_argument('--use_deterministic', action='store_true')
    parser.add_argument('--metrics', nargs='+',
                        default=('avg_cost', 'avg_score', 'surrogate_score', 'avg_violation'))
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    api = wandb.Api()
    sweep = api.sweep(f'unify/long-term-constraints/{args.sweep_id}')

    map_f = partial(get_from_wandb, set=args.set, use_final=args.use_final,
                    use_deterministic=args.use_deterministic, metrics=args.metrics)
    with ThreadPoolExecutor(max_workers=100) as ex:
        values = ex.map(map_f, list(sweep.runs))
    values = list(values)

    variants, strata = zip(*((v[0], v[1]) for v in values))
    variants, strata = set(variants), set(strata)
    data = {m: {v: {s: [] for s in strata} for v in variants} for m in args.metrics}

    for variant, stratum, metrics in values:
        for m in metrics:
            data[m][variant][stratum].append(metrics[m])

    simple_intervals = {m: dict() for m in args.metrics}
    point_estimates = {m: dict() for m in args.metrics}
    for m in args.metrics:
        for v in variants:
            # Compute bootstrap confidence intervals within each stratum
            simple_intervals[m][v] = {s: bootstrap([data[m][v][s]], statistic=np.mean, random_state=42)
                                      for s in strata}
            # Compute point estimates within each stratum, then aggregate across strata
            point_estimates[m][v] = np.mean([np.mean(data[m][v][s]) for s in strata])

    stratified_intervals = {m: dict() for m in args.metrics}
    for m in args.metrics:
        for v in variants:
            # Combine confidence intervals across strata to obtain stratified confidence intervals
            low = np.mean([simple_intervals[m][v][s].confidence_interval.low for s in strata])
            high = np.mean([simple_intervals[m][v][s].confidence_interval.high for s in strata])
            error_margin = (high - low) / 2
            stratified_intervals[m][v] = (point_estimates[m][v] - error_margin, point_estimates[m][v] + error_margin)

    from pprint import pprint
    pprint(stratified_intervals)
