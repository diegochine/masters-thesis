import argparse
import os

import gin
import numpy as np

from utils.training import VARIANTS, RL_ALGOS, train_rl_algo, CONTROLLERS


def get_args_dict():
    """Constructs CLI argument parser, and returns dict of arguments."""
    parser = argparse.ArgumentParser()

    # Main arguments
    parser.add_argument("logdir", type=str, help="Logging directory")
    parser.add_argument("-v", "--variant", type=str, choices=VARIANTS,
                        help="'toy': toy variant of the vpp problem (no battery);"
                             "'standard': standard variant of the vpp problem;"
                             "'cumulative': vpp problem with cumulative constraint on the battery")
    parser.add_argument("-a", "--algo", type=str, choices=RL_ALGOS, default='SACLag',
                        help="Offline RL algorithms to use, 'SACLag'")
    parser.add_argument("-c", "--controller", type=str, choices=CONTROLLERS,
                        help="Type of controller, 'rl' or 'unify'")

    # Additional configs
    parser.add_argument('-sl', '--safety-layer', action='store_true',
                        help="If True, use safety layer to correct unfeasible actions at training time."
                             "Safety Layer is always enabled at testing time to ensure action feasibility.")
    parser.add_argument('--gin', default=None, help='(Optional) path to .gin config file.')

    # Hyper-parameters
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--act-learning-rate", type=float, default=5e-4, help="Actor learning rate")
    parser.add_argument("--crit-learning-rate", type=float, default=5e-4, help="Critic learning rate")
    parser.add_argument("--alpha-learning-rate", type=float, default=1e-3, help="Entropy learning rate")
    parser.add_argument("--lambda-learning-rate", type=float, default=7.5e-4, help="Lagrangian learning rate")
    parser.add_argument('--net-width', type=int, default=192, help="Number of neurons in each layer of the networks.")
    parser.add_argument('--net-layers', type=int, default=2, help="Number of layers of the networks.")
    parser.add_argument('--tau', type=float, default=0.8, help="Tau for polyak averaging updates of target network.")
    parser.add_argument('--target-update-period', type=int, default=5, help="Steps between target networks update.")
    parser.add_argument('--reward-normalization', action='store_true',
                        help='If True, normalizes rewards by their running standard deviation')
    parser.add_argument("--n-instances", type=int, default=1, help="Number of instances the agent is trained on")

    args = vars(parser.parse_args())
    # compute fc_params
    args['fc_params'] = [args['net_width']] * args['net_layers']
    return args


########################################################################################################################


if __name__ == '__main__':

    # NOTE: you should set the logging directory and the method
    args = get_args_dict()

    LOG_DIR = args['logdir']

    # Randomly choose n instances
    np.random.seed(0)
    indexes = np.arange(10000, dtype=np.int32)
    indexes = list(np.random.choice(indexes, size=args['n_instances']))
    print(indexes)

    # Eventually setup wandb logging and gin config
    if 'wandb.key' in os.listdir():
        key = (f := open('wandb.key')).read()
        f.close()
        tags = [args['algo'].lower(), args['controller'].lower()]
        tags += ['safety_layer'] if args['safety_layer'] else []
        tags += list(map(lambda n: str(n), indexes))
        wandb_params = {'key': key,
                        'project': 'thesis',
                        'entity': 'diegochine',
                        'tags': tags,
                        'group': args['variant']}
    else:
        wandb_params = None
    if args['gin'] is not None:
        gin.parse_config_file(args['gin'])

    # Training routine
    train_rl_algo(instances=indexes,
                  noise_std_dev=0.01,
                  wandb_params=wandb_params,
                  log_dir=LOG_DIR,
                  **args)
