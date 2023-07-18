from typing import Tuple

import numpy as np
import pandas as pd
from gurobipy import Model, GRB
from gymnasium.spaces import Box

from envs.safety_layer_vpp_env import SafetyLayerVPPEnv


class StandardVPPEnv(SafetyLayerVPPEnv):
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
                 bound_storage_in: bool = True,
                 wandb_log: bool = True):
        """
        :param predictions: pandas.Dataframe; predicted PV and Load.
        :param c_grid: numpy.array; c_grid values.
        :param shift: numpy.array; shift values.
        :param controller: str; the type of controller, either 'rl' or 'unify'
        :param noise_std_dev: float; the standard deviation of the additive gaussian noise for the realizations.
        :param savepath: string; if not None, the gurobi models are saved to this directory.
        :param use_safety_layer: bool, if True enable safety layer during training.
        :param bound_storage_in: bool; used to switch between enforcing p_storage_in var upper bound via the optimization model
                                or letting the rl agent learn the constraint.
        """

        super().__init__(predictions=predictions,
                         c_grid=c_grid,
                         shift=shift,
                         controller=controller,
                         noise_std_dev=noise_std_dev,
                         savepath=savepath,
                         use_safety_layer=use_safety_layer,
                         wandb_log=wandb_log)
        self._bound_storage_in = bound_storage_in

        # Here we define the observation and action spaces
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.N * 3 + 1,), dtype=np.float32)
        if self.controller == 'rl':
            self.action_space = Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        else:
            self.action_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def _get_observations(self) -> np.array:
        """
        Return predicted pv and load values as a single array.
        :return: tuple(numpy.array, dict); array is pv and load values for the current instance, dict is info
        """

        observations = np.concatenate((self.p_ren_pv_pred / np.max(self.p_ren_pv_pred),
                                       self.tot_cons_pred / np.max(self.tot_cons_pred)),
                                      axis=0)
        one_hot_timestep = np.zeros(shape=(self.N,))
        one_hot_timestep[int(self.timestep)] = 1
        observations = np.concatenate((observations, one_hot_timestep), axis=0)
        if self.storage is not None:  # only add storage for standard variant, not toy
            observations = np.append(observations, self.storage)

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
        self.storage = self.in_cap  # TODO check if this is correct
        self.timestep = 0
        self.cumulative_cost = 0

        self.history = dict(energy_bought=[], energy_sold=[], diesel_power=[],
                            input_storage=[], output_storage=[], storage_capacity=[])

    def _solve_rl(self, action: np.array) -> Tuple[bool, int | float, np.array, float]:
        """
        Solve the optimization model with the greedy heuristic.
        :param action: numpy.array of shape (4, ); the decision variables for each timestep.
        :return: bool, float; True if the model is feasible, False otherwise and a list of cost for each timestep.
        """

        # Check variables initialization
        self.assert_vars_init()

        if not isinstance(action, np.ndarray):
            action = np.array(action)

        if np.any(action < self.action_space.low) or np.any(action > self.action_space.high):
            action = np.clip(action.astype(np.float64), self.action_space.low, self.action_space.high)

        # We follow this convention:
        # action[0] -> input to storage
        # action[1] -> output from storage
        # action[2] -> power sold to the grid
        # action[3] -> power generated from the diesel source

        # Rescale the actions in their feasible ranges
        storage_in, storage_out, grid_in, diesel_power = self.rescale(action)

        # Keep track if the solution is feasible
        feasible = True

        # Shift from Demand Side Energy Management System
        tilde_cons = (self.shift[self.timestep] + self.tot_cons_real[self.timestep])

        # Set the power out from the grid so that the power balance constraint is satisfied
        grid_out = tilde_cons - self.p_ren_pv_real[self.timestep] - storage_out - diesel_power + storage_in + grid_in

        # Compute the cost
        cost = (self.c_grid[self.timestep] * grid_out + self.c_diesel * diesel_power - self.c_grid[
            self.timestep] * grid_in)

        # If the storage constraints are not satisfied or the energy bought is negative then the solution is not
        # feasible. Use safety layer to compute the closest feasible action
        if storage_in > self.cap_max - self.storage or storage_out > self.storage or grid_out < 0:
            # Compute constraints reward. Maximum is 0 (all constraints satisfied) and it is scaled by 0.1.
            constraint_violation = 0.1 * (min(grid_out, 0) +
                                          min(self.storage - storage_out, 0) +
                                          min(self.cap_max - self.storage - storage_in, 0))
            feasible = False
            if self.use_safety_layer:
                feasible_action = self.safety_layer((storage_in, storage_out, grid_in, diesel_power))
                storage_in, storage_out, grid_in, diesel_power = feasible_action
                feasible_action = feasible_action.reshape(-1, 4)
                grid_out = tilde_cons - self.p_ren_pv_real[
                    self.timestep] - storage_out - diesel_power + storage_in + grid_in
            else:
                grid_out = -grid_out  # arbitrary penalty, TODO simple solution but can we do better?
                feasible_action = None
            cost = (self.c_grid[self.timestep] * grid_out + self.c_diesel * diesel_power - self.c_grid[
                self.timestep] * grid_in)
        else:
            constraint_violation = 0
            feasible_action = np.array([[storage_in, storage_out, grid_in, diesel_power]])

        # Update the storage capacity
        old_cap_x = self.storage
        self.storage = self.storage + storage_in - storage_out

        # Check that the constraints are satisfied (only when action is feasible)
        if feasible or self.use_safety_layer:
            self.assert_constraints(diesel_power=diesel_power, grid_in=grid_in, grid_out=grid_out,
                                    tilde_cons=tilde_cons,
                                    old_cap_x=old_cap_x, storage_in=storage_in, storage_out=storage_out)

        return feasible, cost, feasible_action, constraint_violation

    def _solve_unify(self, c_virt: np.array) -> Tuple[bool, int | float, np.array, float]:
        """
        Solve the optimization model with the greedy heuristic.
        :param c_virt: numpy.array of shape (1, ); the virtual costs multiplied to output storage variable.
        :return: tuple (bool, float, np.ndarray, float); feasible flag, cost, action and constraint violation cost.
        """

        # Check variables initialization
        self.assert_vars_init()

        c_virt = np.squeeze(c_virt)

        # Create an optimization model
        mod = Model()

        # build variables and define bounds
        p_diesel = mod.addVar(vtype=GRB.CONTINUOUS, name="p_diesel")
        p_storage_in = mod.addVar(vtype=GRB.CONTINUOUS, name="p_storage_in")
        p_storage_out = mod.addVar(vtype=GRB.CONTINUOUS, name="p_storage_out")
        p_grid_in = mod.addVar(vtype=GRB.CONTINUOUS, name="p_grid_in")
        p_grid_out = mod.addVar(vtype=GRB.CONTINUOUS, name="p_grid_out")
        cap = mod.addVar(vtype=GRB.CONTINUOUS, name="cap")

        # Shift from Demand Side Energy Management System
        tilde_cons = (self.shift[self.timestep] + self.tot_cons_real[self.timestep])

        # Power balance constraint
        mod.addConstr((self.p_ren_pv_real[self.timestep] + p_storage_out + p_grid_out + p_diesel -
                       p_storage_in - p_grid_in == tilde_cons), "Power_balance")

        # Storage cap
        mod.addConstr(cap == self.storage + p_storage_in - p_storage_out)
        mod.addConstr(cap <= self.cap_max)

        mod.addConstr(p_storage_in <= self.cap_max - self.storage)
        mod.addConstr(p_storage_out <= self.storage)

        if self._bound_storage_in:
            mod.addConstr(p_storage_in <= 200)
        mod.addConstr(p_storage_out <= 200)

        # Diesel and grid bounds
        mod.addConstr(p_diesel <= self.p_diesel_max)
        mod.addConstr(p_grid_in <= 600)

        # Objective function
        obf = (self.c_grid[self.timestep] * p_grid_out + self.c_diesel * p_diesel +
               c_virt * p_storage_in - self.c_grid[self.timestep] * p_grid_in)
        mod.setObjective(obf)

        feasible = self.optimize(mod)

        storage_in = mod.getVarByName('p_storage_in').X
        storage_out = mod.getVarByName('p_storage_out').X
        diesel_power = mod.getVarByName('p_diesel').X
        grid_in = mod.getVarByName('p_grid_in').X
        grid_out = mod.getVarByName('p_grid_out').X

        constraint_violation = min(0, 200 - storage_in)

        if constraint_violation < 0:  # storage_in bound constraint has been violated
            if self.use_safety_layer:
                feasible_action = self.safety_layer((storage_in, storage_out, grid_in, diesel_power))
                storage_in, storage_out, grid_in, diesel_power = feasible_action
                feasible_action = feasible_action.reshape(-1, 4)
                grid_out = tilde_cons - self.p_ren_pv_real[
                    self.timestep] - storage_out - diesel_power + storage_in + grid_in
            else:
                grid_out = 1000.
                feasible_action = None
        else:
            assert feasible  # TODO check
            # Update the storage capacitance
            old_cap_x = self.storage
            self.storage = cap.X
            self.assert_constraints(diesel_power=diesel_power, grid_in=grid_in, grid_out=grid_out,
                                    tilde_cons=tilde_cons,
                                    old_cap_x=old_cap_x, storage_in=storage_in, storage_out=storage_out)

            feasible_action = np.array([storage_in, storage_out, grid_in, diesel_power], dtype=np.float64)

            # update history
            for k, v in (('energy_bought', grid_out), ('energy_sold', grid_in), ('diesel_power', diesel_power),
                         ('input_storage', storage_in), ('output_storage', storage_out), ('storage_capacity', old_cap_x)):
                self.history[k].append(v)

        cost = (self.c_grid[self.timestep] * grid_out + self.c_diesel * diesel_power - self.c_grid[
            self.timestep] * grid_in)

        return feasible, cost, feasible_action, constraint_violation

    def assert_constraints(self, diesel_power, grid_in, grid_out, tilde_cons,
                           old_cap_x=None, storage_in=None, storage_out=None):
        """Makes sure constraints are satisfied."""
        assert 0 <= diesel_power <= self.p_diesel_max, f'{diesel_power}'
        assert 0 <= grid_in <= 600, f'{grid_in}'
        assert grid_out >= 0, f'{grid_out}'

        if self.storage is not None:  # full environment with battery
            assert old_cap_x is not None and storage_in is not None and storage_out is not None
            assert self.storage == old_cap_x + storage_in - storage_out, f'{self.storage} == {old_cap_x} + {storage_in} - {storage_out}'
            assert 0 <= self.storage <= self.cap_max, f'{self.storage}'
            assert storage_in <= self.cap_max - old_cap_x, f'{storage_in}'
            assert storage_out <= old_cap_x, f'{storage_out}'
            assert 0 <= storage_in <= 200, f'{storage_in}'
            assert 0 <= storage_out <= 200, f'{storage_out}'
            power_balance = self.p_ren_pv_real[
                                self.timestep] + storage_out + grid_out + diesel_power - storage_in - grid_in
        else:  # simplified environment with no battery
            power_balance = self.p_ren_pv_real[self.timestep] + grid_out + diesel_power - grid_in

        np.testing.assert_almost_equal(power_balance, tilde_cons, decimal=10)

    def assert_vars_init(self, storage=True):
        """Make sure all required attributes are initialized."""
        assert self.mr is not None, "Instance index must be initialized"
        assert self.c_grid is not None, "c_grid must be initialized"
        assert self.shift is not None, "shifts must be initialized before the step function"
        assert self.p_ren_pv_real is not None, "Real PV values must be initialized before the step function"
        assert self.tot_cons_real is not None, "Real Load values must be initialized before the step function"
        if storage:
            assert self.storage is not None, 'Storage variable must be initialized'

    def rescale(self, action, to_network_range=False):
        """Public method for rescaling actions, either from network range (e.g. (-1, 1)) to env range or vice versa.
        Args:
            action: np.array, action to be rescaled.
            to_network_range: (Optional) if True, rescales from env range to network range; otherwise performs the opposite
                rescaling. Defaults to False.
        """
        net_lb = np.full(4, -1.)
        net_ub = np.ones(4)
        env_lb = np.zeros(4)
        env_ub = np.array([200, 200, 600, self.p_diesel_max])

        if to_network_range:
            std = (action - env_lb) / (env_ub - env_lb)
            action = std * (net_ub - net_lb) + net_lb
        else:
            std = (action - net_lb) / (net_ub - net_lb)
            action = std * (env_ub - env_lb) + env_lb
        return action

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
        if self.controller == 'rl':
            feasible, cost, actual_action, constraint_violation = self._solve_rl(action)
        else:  # controller = unify
            feasible, cost, actual_action, constraint_violation = self._solve_unify(action)

        if feasible or self.use_safety_layer:
            self.cumulative_cost += cost

        observations, _ = self._get_observations()

        # Update the timestep
        self.timestep += 1
        assert self.timestep <= self.N, f"Timestep cannot be greater than {self.N}"
        terminated, truncated = False, False
        if not feasible and not self.use_safety_layer:
            reward = self._min_rewards[self.timestep - 1]
            truncated = True
        else:
            reward = -cost
            terminated = (self.timestep == self.N)

        if terminated or truncated:
            self.log()

        return observations, reward, terminated, truncated, {'feasible': feasible,
                                                             'action': actual_action,
                                                             'sl_usage': self.sl_counter / self.timestep,
                                                             'constraint_violation': constraint_violation}
