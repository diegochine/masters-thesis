import os

import gin
import gymnasium
import numpy as np
import pandas as pd
from typing import Union, List
import wandb
import tqdm
from pyagents import networks
from keras.optimizers import Adam

from envs import ToyVPPEnv, StandardVPPEnv
from agents import SACLagEMS
from utils.online_heuristic import compute_real_cost

########################################################################################################################

TIMESTEP_IN_A_DAY = 96

VARIANTS = ['toy', 'standard']
CONTROLLERS = ['rl', 'unify']
RL_ALGOS = ['saclag']

wandb_running = lambda: wandb.run is not None


########################################################################################################################

def make_env(variant, instances, controller: str, noise_std_dev: Union[float, int] = 0.01,
             safety_layer: bool = False):
    # FIXME: the filepath should not be hardcoded
    predictions_filepath = os.path.join('data', 'Dataset10k.csv')
    prices_filepath = os.path.join('data', 'gmePrices.npy')
    shifts_filepath = os.path.join('data', 'optShift.npy')

    # Load data from file
    # Check that all the required files exist
    assert os.path.isfile(predictions_filepath), f"{predictions_filepath} does not exist"
    assert os.path.isfile(prices_filepath), f"{prices_filepath} does not exist"
    assert os.path.isfile(shifts_filepath), f"{shifts_filepath} does not exist"
    predictions = pd.read_csv(predictions_filepath)
    shift = np.load(shifts_filepath)
    c_grid = np.load(prices_filepath)

    # Split between training and test
    if isinstance(instances, float):
        split_index = int(len(predictions) * (1 - instances))
        train_predictions = predictions[:split_index]
    elif isinstance(instances, list):
        train_predictions = predictions.iloc[instances]
    else:
        raise Exception("test_split must be list of int or float")

    # Create environment
    if variant == 'toy':
        env = ToyVPPEnv(predictions=train_predictions,
                        shift=shift,
                        c_grid=c_grid,
                        controller=controller,
                        noise_std_dev=noise_std_dev,
                        savepath=None,
                        use_safety_layer=safety_layer)
    elif variant == 'standard':
        env = StandardVPPEnv(predictions=train_predictions,
                             shift=shift,
                             c_grid=c_grid,
                             controller=controller,
                             noise_std_dev=noise_std_dev,
                             savepath=None,
                             use_safety_layer=safety_layer,
                             bound_storage_in=False)  # FIXME should not be hardcoded
    else:
        raise ValueError(f'Variant name must be in {VARIANTS}')
    print(f'Selected variant: {variant}')
    env = gymnasium.vector.SyncVectorEnv([lambda: env])
    env = gymnasium.wrappers.RecordEpisodeStatistics(env)

    return env


########################################################################################################################


def train_loop(agent, env, num_epochs, batch_size,
               rollout_steps=1, train_steps=1, test_every=200, store_feasible=False, test_env=None):
    k = 1
    episode = 0
    # Test untrained agent
    best_score, constraints_rews = test_agent(agent, test_env, render_plots=False)
    if wandb_running():
        wandb.log({'test/score': best_score, 'test/constraint_rewards': constraints_rews})
    num_envs = getattr(env, "num_envs", 1)

    # Main loop
    s_t, _ = env.reset()  # pyagents wants first dim to be batch dim
    for epoch in (pbar := tqdm.trange(0, num_epochs, train_steps, desc='TRAINING')):

        # Env interactions
        for _ in range(rollout_steps):
            agent_out = agent.act(s_t)
            a_t, lp_t = agent_out.actions, agent_out.logprobs
            s_tp1, r_t, terminated, truncated, info = env.step(a_t[0])

            if 'final_info' in info:
                envs_over = np.logical_or(terminated, truncated)
                s_tp1_with_final = np.where(envs_over, np.stack(info['final_observation']), s_tp1)
                # FIXME when using n_envs>1, costs may be wrong for unterminated sub-environments
                all_costs = [info['final_info'][i].get('constraint_violation', 0.0) for i in range(agent.num_envs)]
                r_t = np.column_stack([r_t, all_costs])
                agent.remember(state=s_t,
                               action=a_t,
                               reward=r_t,
                               next_state=s_tp1_with_final,
                               done=terminated)
            else:
                r_t = np.column_stack([r_t, info['constraint_violation']])
                agent.remember(state=s_t,
                               action=a_t,
                               reward=r_t,
                               next_state=s_tp1,
                               done=terminated)

            if wandb_running():
                if 'episode' in info:
                    episode += 1
                    wandb.log({'episode': episode,
                               'train/score': info['episode']['r'],
                               'train/length': info['episode']['l']})
            s_t = s_tp1
        # Training
        for _ in range(train_steps):
            loss_dict = agent.train(batch_size)
            pbar.set_postfix(loss_dict)

        # Testing
        if test_env is not None and epoch > k * test_every:
            pbar.set_description('TESTING')
            score, constraints_rews = test_agent(agent, test_env, render_plots=False)
            if score > best_score:
                best_score = score
                agent.save(ver=k)
            k += 1
            loss_dict['test/score'] = score
            if wandb_running():
                wandb.log({'test/score': score, 'test/constraint_rewards': constraints_rews})
            pbar.set_description(f'[EVAL SCORE: {score:4.0f}] TRAINING')

    return agent


########################################################################################################################

def test_agent(agent, test_env, render_plots=True, save_path=None):
    done = False
    s_t = test_env.reset()[0].reshape(1, -1)  # pyagents wants first dim to be batch dim
    scores = []
    all_actions = []
    constraint_rewards = []

    # Perform an episode
    while not done:
        agent_out = agent.act(s_t)
        a_t, lp_t = agent_out.actions, agent_out.logprobs
        s_tp1, r_t, terminated, truncated, info = test_env.step(a_t[0])
        done = terminated or truncated
        if 'action' in info:  # get SL-corrected action
            a_t = info['action'][0]
        s_tp1 = s_tp1.reshape(1, -1)
        all_actions.append(np.squeeze(a_t))
        constraint_rewards.append(info['constraint_violation'])
        scores.append(r_t)
        s_t = s_tp1

    if render_plots:
        compute_real_cost(instance_idx=test_env.env.envs[0].mr,
                          predictions_filepath=os.path.join('data', 'Dataset10k.csv'),
                          shifts_filepath=os.path.join('data', 'optShift.npy'),
                          prices_filepath=os.path.join('data', 'gmePrices.npy'),
                          decision_variables=np.array(all_actions),
                          display=False,
                          savepath=save_path,
                          wandb_log=agent.is_logging)
    return np.sum(scores), np.mean(constraint_rewards)


########################################################################################################################

@gin.configurable
def train_rl_algo(variant: str = None,
                  controller: str = None,
                  algo: str = 'SACLag',
                  safety_layer: bool = False,
                  instances: Union[float, List[int]] = 0.25,
                  epochs: int = 1000,
                  noise_std_dev: Union[float, int] = 0.01,
                  batch_size: int = 100,
                  crit_learning_rate: float = 7e-4,
                  act_learning_rate: float = 7e-4,
                  alpha_learning_rate: float = 7e-4,
                  lambda_learning_rate: float = 1e-3,
                  fc_params: list = [256, 256],
                  rollout_steps: int = 1,
                  train_steps: int = 1,
                  log_dir: str = 'output',
                  wandb_params: dict = None,
                  test_every: int = 200,
                  n_envs: int = 1,
                  tau: float = 0.001,
                  target_update_period: int = 10,
                  **kwargs):
    """
    Training routine.
    :param variant: string; choose among one of the available methods.
    :param instances: float or list of int; fraction or indexes of the instances to be used for test.
    :param epochs: int; number of training epochs.
    :param noise_std_dev: float; standard deviation for the additive gaussian noise.
    :param batch_size: int; batch size.
    :return:
    """

    env = make_env(variant, instances, controller, noise_std_dev, safety_layer=safety_layer)
    test_env = make_env(variant, instances, controller, noise_std_dev,
                        safety_layer=(controller == 'rl'))  # turn on safety layer during testing only for rl approaches

    # Get observation and action spaces
    obs_space = getattr(env, 'single_observation_space', env.observation_space)
    act_space = getattr(env, 'single_action_space', env.action_space)
    state_shape = obs_space.shape
    action_shape = act_space.shape
    bounds = (-1.0, 1.0)  # TODO okay for unify also?
    a_net = networks.PolicyNetwork(state_shape, action_shape, fc_params=fc_params,
                                   output='gaussian', bounds=bounds, activation='relu',
                                   out_params={'state_dependent_std': True,
                                               'mean_activation': None})

    log_dict = dict(act_learning_rate=act_learning_rate,
                    crit_learning_rate=crit_learning_rate,
                    alpha_learning_rate=alpha_learning_rate,
                    lambda_learning_rate=lambda_learning_rate,
                    num_epochs=epochs, batch_size=batch_size,
                    rollout_steps=rollout_steps, train_steps=train_steps, n_envs=n_envs)

    if algo == 'saclag':
        reward_shape = (2,)
        q_nets = networks.ExtendedQNetwork(state_shape=state_shape, action_shape=action_shape,
                                           reward_shape=reward_shape,
                                           n_critics=2)
        a_opt = Adam(learning_rate=act_learning_rate)
        c_opt = Adam(learning_rate=crit_learning_rate)
        alpha_opt = Adam(learning_rate=alpha_learning_rate)
        lambda_opt = Adam(learning_rate=lambda_learning_rate)

        agent = SACLagEMS(state_shape, action_shape, buffer='uniform', gamma=1.0, reward_shape=reward_shape,
                          actor=a_net, critics=q_nets, tau=tau, target_update_period=target_update_period,
                          actor_opt=a_opt, critic_opt=c_opt, alpha_opt=alpha_opt, lambda_opt=lambda_opt,
                          wandb_params=wandb_params, save_dir=log_dir, log_dict=log_dict)

    else:
        raise Exception("algo is SACLag")
    agent.init(envs=env, rollout_steps=rollout_steps, min_memories=2000)
    agent = train_loop(agent=agent, env=env, num_epochs=epochs, batch_size=batch_size, rollout_steps=rollout_steps,
                       train_steps=train_steps, test_every=test_every, test_env=test_env)
    test_agent(agent, test_env, render_plots=False)
    agent.save('_final')
