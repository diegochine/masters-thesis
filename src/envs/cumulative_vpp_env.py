from typing import Tuple

import numpy as np
import pandas as pd
import wandb
from gymnasium.spaces import Box

from src.envs.standard_vpp_env import StandardVPPEnv


class CumulativeVPPEnv(StandardVPPEnv):
    """
    Gym environment for the Markovian version of the VPP optimization model.
    """

    metadata = {
        "render.modes": ["ascii"]
    }

    def __init__(self,
                 predictions: pd.DataFrame,
                 c_grid: np.ndarray,
                 shift: np.ndarray,
                 controller: str,
                 noise_std_dev: float = 0.02,
                 savepath: str = None,
                 use_safety_layer: bool = False,
                 wandb_run: wandb.sdk.wandb_run.Run | None = None,
                 cumulative_storage_bound: float = 0.5,
                 **kwargs):
        """
        :param predictions: pandas.Dataframe; predicted PV and Load.
        :param c_grid: numpy.array; c_grid values.
        :param shift: numpy.array; shift values.
        :param controller: str; the type of controller, either 'rl' or 'unify'
        :param noise_std_dev: float; the standard deviation of the additive gaussian noise for the realizations.
        :param savepath: string; if not None, the gurobi models are saved to this directory.
        :param use_safety_layer: bool, if True enable safety layer during training.
        :param cumulative_storage_bound: float; the coefficient of the cumulative constraint.
               The cumulative constraint states that on average storage capacity should be >= cumulative_storage_bound.
        """

        super().__init__(predictions=predictions,
                         c_grid=c_grid,
                         shift=shift,
                         controller=controller,
                         noise_std_dev=noise_std_dev,
                         savepath=savepath,
                         use_safety_layer=use_safety_layer,
                         bound_storage_in=True,
                         wandb_run=wandb_run,
                         **kwargs)

        # Here we define the observation and action spaces
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.N * 3 + 1,), dtype=np.float32)
        if self.controller == 'rl':
            self.action_space = Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        else:
            self.action_space = Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        # Coefficient of the cumulative constraint,
        # i.e. on average storage capacity should be >= cumulative_storage_bound
        assert 0 <= cumulative_storage_bound <= 1, f'cumulative_storage_bound must be in [0, 1], received: {cumulative_storage_bound}'
        self.cumulative_storage_bound = cumulative_storage_bound

    def _solve_rl(self, action: np.array) -> Tuple[bool, int | float, np.array, float]:
        """
        Solve the optimization model with the greedy heuristic.
        :param action: numpy.array of shape (4, ); the decision variables for each timestep.
        :return: bool, float; True if the model is feasible, False otherwise and a list of cost for each timestep.
        """
        raise NotImplementedError()

    def step(self, action: np.array) -> Tuple[np.array, np.ndarray, bool, bool, dict]:
        """
        This is a step performed in the environment: the virtual costs are set by the agent.py and then the total cost
        (the reward) is computed.
        :param action: numpy.array of shape (num_timesteps, ); virtual costs for each timestep.
        :return: numpy.array, float, boolean, boolean dict; the observations, the reward,
                                    a boolean that is True if the episode has terminated,
                                    a boolean that is True if the episode has been truncated (e.g. unfeasible action),
                                    additional information.
        """
        observations, reward, terminated, truncated, info = super().step(action)
        if self.timestep == self.N:
            # As we need constraint in the form E[Z] <= b, we consider average free storage insted of occupied storage
            # i.e. E[cap(t)] >= c can be rewritten as E[cap_max - cap(t)] <= 1 - c
            free_storage = self.cap_max - np.mean(self.history['storage_capacity'])
            # storage is in [0, cap_max] but c is in [0, 1]
            constraint_cost = free_storage  # max(0., free_storage - ((1 - self.cumulative_storage_bound) * self.cap_max))
        else:
            constraint_cost = 0.
        reward = np.array([reward[0], constraint_cost])

        return observations, reward, terminated, truncated, info
