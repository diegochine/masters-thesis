from typing import Tuple

import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from envs.standard_vpp_env import StandardVPPEnv


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
                 wandb_log: bool = True,
                 cumulative_storage_bound: float = 0.5):
        """
        :param predictions: pandas.Dataframe; predicted PV and Load.
        :param c_grid: numpy.array; c_grid values.
        :param shift: numpy.array; shift values.
        :param controller: str; the type of controller, either 'rl' or 'unify'
        :param noise_std_dev: float; the standard deviation of the additive gaussian noise for the realizations.
        :param savepath: string; if not None, the gurobi models are saved to this directory.
        :param use_safety_layer: bool, if True enable safety layer during training.
        :param cumulative_storage_bound: float; the coefficient of the cumulative constraint.
        """

        super().__init__(predictions=predictions,
                         c_grid=c_grid,
                         shift=shift,
                         controller=controller,
                         noise_std_dev=noise_std_dev,
                         savepath=savepath,
                         use_safety_layer=use_safety_layer,
                         bound_storage_in=True,
                         wandb_log=wandb_log)

        # Here we define the observation and action spaces
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.N * 3 + 1,), dtype=np.float32)
        if self.controller == 'rl':
            self.action_space = Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        else:
            self.action_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        # Coefficient of the cumulative constraint,
        # i.e. on average storage capacity should be >= cumulative_storage_bound
        self.cumulative_storage_bound = cumulative_storage_bound

    def _solve_rl(self, action: np.array) -> Tuple[bool, int | float, np.array, float]:
        """
        Solve the optimization model with the greedy heuristic.
        :param action: numpy.array of shape (4, ); the decision variables for each timestep.
        :return: bool, float; True if the model is feasible, False otherwise and a list of cost for each timestep.
        """
        raise NotImplementedError()

    def step(self, action: np.array) -> Tuple[np.array, int | float, bool, bool, dict]:
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
            violation = np.mean(self.history['storage_capacity']) - (self.cap_max * self.cumulative_storage_bound)
            info['constraint_violation'] = min(0., violation)
        else:
            info['constraint_violation'] = 0.

        return observations, reward, terminated, truncated, info
