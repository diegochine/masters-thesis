from datetime import datetime, timedelta
from typing import List, Any

import numpy as np
import pandas as pd
from gymnasium import Env
import gurobipy
from gurobipy import GRB
from gymnasium.spaces import Box
from tabulate import tabulate
import wandb


########################################################################################################################


class VPPEnv(Env):
    """
    Gym environment for the VPP optimization model.
    """

    # Reward for unfeasible actions
    MIN_REWARD = -2000
    # Number of timesteps in one day
    N = 96

    ACTIONS = ['energy_bought', 'energy_sold', 'diesel_power',
               'input_storage', 'output_storage', 'storage_capacity',
               'c_virt_in', 'c_virt_out']

    # This is a gym.Env variable that is required by garage for rendering
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
                 in_cap: int = 0,
                 wandb_run: wandb.sdk.wandb_run.Run | None = None,
                 fixed_noise: bool = False):
        """
        :param predictions: pandas.Dataframe; predicted PV and Load.
        :param c_grid: numpy.array; c_grid values.
        :param shift: numpy.array; shift values.
        :param controller: str; the type of controller, either 'rl' or 'unify'
        :param noise_std_dev: float; the standard deviation of the additive gaussian noise for the realizations.
        :param savepath: string; if not None, the gurobi models are saved to this directory.
        :param use_safety_layer: bool; if True, use safety layer during training.
        :param in_cap: int; the capacity of the storage at the beginning of the episode.
        :param wandb_run: wandb.sdk.wandb_run.Run; wandb run object for logging.
        :param fixed_noise: bool; if True, the noise is sampled only once, otherwise it is sampled at each reset.
                Used for evaluation env.
        """

        # Counter of safety layer usage
        self.sl_counter = 0

        # Controller type
        assert controller in ('rl', 'unify'), f"controller parameter must be either 'rl' or 'unify'," \
                                              f"received: {controller}"
        self.controller = controller

        # Standard deviation of the additive gaussian noise
        self.noise_std_dev = noise_std_dev

        # These are variables related to the optimization model
        self.predictions = predictions
        self.predictions = self.instances_preprocessing(self.predictions)
        self.c_grid = c_grid
        self.shift = shift
        self.cap_max = 1000
        self.in_cap = in_cap
        self.c_diesel = 0.054
        self.p_diesel_max = 1200

        # We randomly choose an instance
        self.mr = np.random.choice(self.predictions.index)

        if fixed_noise:
            assert len(self.predictions) == 1, f'fixed_noise can be used only with one instance'
            rng = np.random.default_rng(self.mr)
            self.noise = (rng.normal(0, self.noise_std_dev, self.N),
                          rng.normal(0, self.noise_std_dev, self.N))
        else:
            self.noise = None

        self.savepath = savepath
        self.use_safety_layer = use_safety_layer
        self._min_rewards = [self.MIN_REWARD * 0.99 ** i for i in range(self.N)]
        self._create_instance_variables()

        # wandb Run object for logging and episode step tracker
        self._wandb_run = wandb_run
        self._episode_step = 0

        # needed by torchrl
        self.reward_space = Box(low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf]),
                                shape=(2,), dtype=np.float32)

    @staticmethod
    def instances_preprocessing(instances: pd.DataFrame) -> pd.DataFrame:
        """
        Convert PV and Load values from string to float.
        :param instances: pandas.Dataframe; PV and Load for each timestep and for every instance.
        :return: pandas.Dataframe; the same as the input dataframe but with float values instead of string.
        """

        assert 'PV(kW)' in instances.keys(), "PV(kW) must be in the dataframe columns"
        assert 'Load(kW)' in instances.keys(), "Load(kW) must be in the dataframe columns"

        # Instances pv from file
        instances.loc[:, 'PV(kW)'] = instances['PV(kW)'].map(lambda entry: entry[1:-1].split())
        instances.loc[:, 'PV(kW)'] = instances['PV(kW)'].map(lambda entry: list(np.float_(entry)))

        # Instances load from file
        instances.loc[:, 'Load(kW)'] = instances['Load(kW)'].map(lambda entry: entry[1:-1].split())
        instances.loc[:, 'Load(kW)'] = instances['Load(kW)'].map(lambda entry: list(np.float_(entry)))

        return instances

    @staticmethod
    def optimize(mod: gurobipy.Model) -> bool:
        """
        Solves an optimization model.
        :param mod: gurobipy.Model; the optimization model to be solved.
        :return: bool; True if the optimal solution is found, False otherwise.
        """

        mod.setParam('OutputFlag', 0)
        mod.optimize()
        status = mod.status
        if status == GRB.Status.UNBOUNDED:
            print('\nThe model is unbounded')
            return False
        elif status == GRB.Status.INFEASIBLE:
            print('\nThe model is infeasible')
            return False
        elif status == GRB.Status.INF_OR_UNBD:
            print('\nThe model is either infeasible or unbounded')
            return False

        if status != GRB.Status.OPTIMAL:
            print('\nOptimization was stopped with status %d' % status)
            return False

        return True

    @staticmethod
    def timestamps_headers(num_timeunits: int) -> List[str]:
        """
        Given a number of timeunits (in minutes), it provides a string representation of each timeunit.
        For example, if num_timeunits=96, the result is [00:00, 00:15, 00:30, ...].
        :param num_timeunits: int; the number of timeunits in a day.
        :return: list of string; list of timeunits.
        """

        start_time = datetime.strptime('00:00', '%H:%M')
        timeunit = 24 * 60 / num_timeunits
        timestamps = [start_time + idx * timedelta(minutes=timeunit) for idx in range(num_timeunits)]
        timestamps = ['{:02d}:{:02d}'.format(timestamp.hour, timestamp.minute) for timestamp in timestamps]

        return timestamps

    def _create_instance_variables(self):
        """
        Create predicted and real, PV and Load for the current instance.
        :return:
        """

        assert self.mr is not None, "Instance index must be initialized"

        # predicted PV for the current instance
        self.p_ren_pv_pred = self.predictions['PV(kW)'][self.mr]
        self.p_ren_pv_pred = np.asarray(self.p_ren_pv_pred)

        # predicted Load for the current instance
        self.tot_cons_pred = self.predictions['Load(kW)'][self.mr]
        self.tot_cons_pred = np.asarray(self.tot_cons_pred)

        if self.noise is None:
            if self._np_random is not None:
                noise_pv = self._np_random.normal(0, self.noise_std_dev, self.N)
                noise_load = self._np_random.normal(0, self.noise_std_dev, self.N)
            else:
                noise_pv = np.random.normal(0, self.noise_std_dev, self.N)
                noise_load = np.random.normal(0, self.noise_std_dev, self.N)
        else:
            noise_pv, noise_load = self.noise

        # The real PV for the current instance is computed adding noise to the predictions
        self.p_ren_pv_real = self.p_ren_pv_pred + self.p_ren_pv_pred * noise_pv

        # The real Load for the current instance is computed adding noise to the predictions
        self.tot_cons_real = self.tot_cons_pred + self.tot_cons_pred * noise_load

        # Reset the timestep
        self.timestep = 0

        # Reset the cumulative cost
        self.cumulative_cost = 0

        # Reset SL counter
        self.sl_counter = 0

        # Reset the actions' history
        self.history = {a: [] for a in self.ACTIONS}

    def step(self, action: np.array):
        """
        Step function of the Gym environment.
        :param action: numpy.array; agent.py's action.
        :return:
        """
        raise NotImplementedError()

    def _clear(self):
        """
        Clear all the instance variables.
        """
        raise NotImplementedError('Subclasses must implement _clear() method')

    def _get_observations(self) -> tuple[np.array, dict[str, Any]]:
        """
        Get the observations for the agent.
        :return: tuple(numpy.array, dict); array is pv and load values for the current instance, dict is info
        """
        raise NotImplementedError('Subclasses must implement _get_observations() method')

    def reset(self,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.array, dict[str, Any]]:
        """
        When we reset the environment we randomly choose another instance and we clear all the instance variables.
        :return: tuple(numpy.array, dict); array is pv and load values for the current instance, dict is info
        """

        super().reset(seed=seed)

        self._clear()

        # We randomly choose an instance
        self.mr = np.random.choice(self.predictions.index)
        self._create_instance_variables()
        return self._get_observations()

    def render(self, mode: str = 'ascii'):
        """
        Simple rendering of the environment.
        :return:
        """

        timestamps = self.timestamps_headers(self.N)
        print('\nPredicted PV(kW)')
        print(tabulate(np.expand_dims(self.p_ren_pv_pred, axis=0), headers=timestamps, tablefmt='pretty'))
        print('\nPredicted Load(kW)')
        print(tabulate(np.expand_dims(self.tot_cons_pred, axis=0), headers=timestamps, tablefmt='pretty'))
        print('\nReal PV(kW)')
        print(tabulate(np.expand_dims(self.p_ren_pv_real, axis=0), headers=timestamps, tablefmt='pretty'))
        print('\nReal Load(kW)')
        print(tabulate(np.expand_dims(self.tot_cons_real, axis=0), headers=timestamps, tablefmt='pretty'))

    def close(self):
        """
        Close the environment.
        :return:
        """
        pass

    def log(self, **kwargs):
        """
        Logs training info using wandb.
        :return:
        """
        if self._wandb_run is not None:
            actions_log = {f'train/{k}': np.array(v).reshape(-1, 1) for k, v in self.history.items()}
            self._wandb_run.log({'train_episode': self._episode_step, **actions_log})
            self._episode_step += 1
