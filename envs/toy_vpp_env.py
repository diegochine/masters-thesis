from typing import Tuple, Union, Dict

import numpy as np
import pandas as pd
from gurobipy import Model, GRB
from gymnasium.spaces import Box

from envs.standard_vpp_env import StandardVPPEnv


class ToyVPPEnv(StandardVPPEnv):
    """
    Gym environment for a simplified version of the (Markovian) VPP optimization model.
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
                 use_safety_layer: bool = False):
        """
        :param predictions: pandas.Dataframe; predicted PV and Load.
        :param c_grid: numpy.array; c_grid values.
        :param shift: numpy.array; shift values.
        :param controller: str; the type of controller, either 'rl' or 'unify'
        :param noise_std_dev: float; the standard deviation of the additive gaussian noise for the realizations.
        :param savepath: string; if not None, the gurobi models are saved to this directory.
        :param use_safety_layer: bool, if True enable safety layer
        """
        if controller == 'unify':
            # discard c_grid data and synthetically generate values
            self.synthetic_data = np.random.normal(np.mean(c_grid), np.var(c_grid), (self.N,))
            c_grid = self.synthetic_data ** 2 - (0.5 * self.synthetic_data)

        super().__init__(predictions=predictions,
                         c_grid=c_grid,
                         shift=shift,
                         controller=controller,
                         noise_std_dev=noise_std_dev,
                         savepath=savepath,
                         use_safety_layer=use_safety_layer)

        # Override spaces (shapes) due to the absence of the battery
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.N * 3,), dtype=np.float32)
        if self.controller == 'rl':
            self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        else:
            # +1 for synthetic data
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.N * 3 + 1,), dtype=np.float32)
            self.action_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def _get_observations(self) -> Tuple[np.array, Dict[str, float]]:
        """
        Return predicted pv and load values as a single array.
        :return: tuple(numpy.array, dict); array is pv and load values for the current instance, dict is info
        """
        observations, _ = super()._get_observations()

        if self.controller == 'unify':  # also add syntethic data to observation
            observations = np.concatenate((observations, self.synthetic_data[[self.timestep]]), axis=0)

        return observations, dict(constraint_violation=0.)

    def _clear(self):
        """
        Clear all the instance dependent variables.
        :return:
        """
        self.p_ren_pv_pred = None
        self.p_ren_pv_real = None
        self.tot_cons_pred = None
        self.tot_cons_real = None
        self.mr = None
        self.timestep = 0
        self.cumulative_cost = 0

        self.energy_bought = []
        self.energy_sold = []
        self.diesel_power = []

        self.storage = None  # Not considering the battery in this simplified environment

    def _solve_rl(self, action: np.array) -> Tuple[bool, Union[int, float], np.array, float]:
        """
        Solve the optimization model with the greedy heuristic.
        :param action: numpy.array of shape (2, ); the decision variables for each timestep.
        :return: bool, float; True if the model is feasible, False otherwise and a list of cost for each timestep.
        """

        # Check variables initialization
        self.assert_vars_init(storage=False)

        if not isinstance(action, np.ndarray):
            action = np.array(action)

        if np.any(action < self.action_space.low) or np.any(action > self.action_space.high):
            action = np.clip(action.astype(np.float64), self.action_space.low, self.action_space.high)

        # We follow this convention:
        # action[1] -> power sold to the grid
        # action[2] -> power generated from the diesel source

        # Rescale the actions in their feasible ranges
        grid_in, diesel_power = self.rescale(action)

        # Keep track if the solution is feasible
        feasible = True

        # Shift from Demand Side Energy Management System
        tilde_cons = (self.shift[self.timestep] + self.tot_cons_real[self.timestep])

        # Set the power out from the grid so that the power balance constraint is satisfied
        grid_out = tilde_cons - self.p_ren_pv_real[self.timestep] - diesel_power + grid_in

        # Compute the cost
        cost = self.c_grid[self.timestep] * grid_out + \
               self.c_diesel * diesel_power - \
               self.c_grid[self.timestep] * grid_in

        # If the energy bought is negative then the solution is not feasible.
        # Use safety layer to compute the closest feasible action
        if grid_out < 0:
            # Compute constraints reward. Maximum is 0 (all constraints satisfied) and it is scaled by 0.1.
            constraint_violation = 0.1 * grid_out
            feasible = False
            if self.use_safety_layer:
                feasible_action = self.safety_layer((grid_in, diesel_power))
                grid_in, diesel_power = feasible_action
                grid_out = tilde_cons - self.p_ren_pv_real[self.timestep] - diesel_power + grid_in
            else:
                grid_out = 1000  # arbitrary penalty, # TODO simple solution but can we do better?
                feasible_action = None
            cost = (self.c_grid[self.timestep] * grid_out + self.c_diesel * diesel_power - self.c_grid[
                self.timestep] * grid_in)
        else:
            constraint_violation = 0
            feasible_action = np.array([grid_in, diesel_power])

        # Check that the constraints are satisfied
        if feasible or self.use_safety_layer:
            self.assert_constraints(diesel_power, grid_in, grid_out, tilde_cons)

        return feasible, cost, feasible_action, constraint_violation

    def _solve_unify(self, c_virt: np.array) -> Tuple[bool, int | float, np.ndarray, float]:
        """
        Solve the optimization model with the greedy heuristic.
        :param c_virt: numpy.array of shape (1, ); the virtual costs, i.e. c_grid for the toy problem.
        :return: tuple (bool, float, np.ndarray, float); feasible flag, cost, action and constraint violation cost.
        """
        # Check variables initialization
        self.assert_vars_init(storage=False)

        c_virt = np.squeeze(c_virt)

        # Create an optimization model
        mod = Model()

        # build variables and define bounds
        p_diesel = mod.addVar(vtype=GRB.CONTINUOUS, name="p_diesel")
        p_grid_in = mod.addVar(vtype=GRB.CONTINUOUS, name="p_grid_in")
        p_grid_out = mod.addVar(vtype=GRB.CONTINUOUS, name="p_grid_out")

        # Shift from Demand Side Energy Management System
        tilde_cons = (self.shift[self.timestep] + self.tot_cons_real[self.timestep])

        # Power balance constraint
        mod.addConstr((self.p_ren_pv_real[self.timestep] + p_grid_out + p_diesel - p_grid_in == tilde_cons), "Power_balance")

        # Diesel and grid bounds
        mod.addConstr(p_diesel <= self.p_diesel_max)
        mod.addConstr(p_grid_in <= 600)

        # Objective function
        obf = (c_virt * p_grid_out + self.c_diesel * p_diesel - c_virt * p_grid_in)
        mod.setObjective(obf)

        feasible = self.optimize(mod)

        diesel_power = mod.getVarByName('p_diesel').X
        grid_in = mod.getVarByName('p_grid_in').X
        grid_out = mod.getVarByName('p_grid_out').X

        action = np.array([grid_in, diesel_power], dtype=np.float64)
        cost = c_virt * grid_out + self.c_diesel * diesel_power - c_virt * grid_in

        return feasible, cost, action, (0. if feasible else -100.)

    def rescale(self, action, to_network_range=False):
        """Public method for rescaling actions, either from network range (e.g. (-1, 1)) to env range or vice versa.
        Args:
            action: np.array, action to be rescaled.
            to_network_range: (Optional) if True, rescales from env range to network range; otherwise performs the opposite
                rescaling. Defaults to False.
        """
        net_lb = np.full(self.action_space.shape, -1.)
        net_ub = np.ones(self.action_space.shape)
        env_lb = np.zeros(self.action_space.shape)
        env_ub = np.array([600, self.p_diesel_max])

        if to_network_range:
            std = (action - env_lb) / (env_ub - env_lb)
            action = std * (net_ub - net_lb) + net_lb
        else:
            std = (action - net_lb) / (net_ub - net_lb)
            action = std * (env_ub - env_lb) + env_lb
        return action
