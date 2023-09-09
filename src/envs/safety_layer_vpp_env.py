import numpy as np
import pandas as pd
import wandb
from gurobipy import Model, GRB

from src import envs


class SafetyLayerVPPEnv(envs.VPPEnv):
    """
    Wrapper environment providing the safety layer for the Markovian version of the VPP optimization model.
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
                 wandb_run: wandb.sdk.wandb_run.Run | None = None):
        """
        :param predictions: pandas.Dataframe; predicted PV and Load.
        :param c_grid: numpy.array; c_grid values.
        :param shift: numpy.array; shift values.
        :param controller: str; the type of controller, either 'rl' or 'unify'
        :param noise_std_dev: float; the standard deviation of the additive gaussian noise for the realizations.
        :param savepath: string; if not None, the gurobi models are saved to this directory.
        :param use_safety_layer: bool, if True enable safety layer
        """

        super().__init__(predictions=predictions,
                         c_grid=c_grid,
                         shift=shift,
                         controller=controller,
                         noise_std_dev=noise_std_dev,
                         savepath=savepath,
                         use_safety_layer=use_safety_layer,
                         wandb_run=wandb_run)

    def safety_layer(self, action, eps=0.5):  # TODO make external function
        """
        Find a feasible action using safety layer.
        :param action: numpy.array of shape (4, ) or (2,); the decision variables for each timestep
        :param eps: float, epsilon used to limit ranges (in both directions) for numerical stability
        :return: numpy.array of shape (4, ); closest feasible action
        """
        self.sl_counter += 1

        tilde_cons = self.shift[self.timestep] + self.tot_cons_real[self.timestep]
        tilde_ren = self.p_ren_pv_real[self.timestep]

        # create optimization model, make variables and set constraints
        mod = Model()

        grid_in_hat = mod.addVar(vtype=GRB.SEMICONT, lb=eps, ub=600 - eps,
                                 name="grid_in_hat")
        diesel_power_hat = mod.addVar(vtype=GRB.SEMICONT, lb=eps, ub=self.p_diesel_max - eps,
                                      name="diesel_power_hat")
        grid_out = mod.addVar(vtype=GRB.CONTINUOUS, name="grid_out")

        if isinstance(self, envs.StandardVPPEnv):  # standard variant
            storage_in_hat = mod.addVar(vtype=GRB.SEMICONT, lb=eps, ub=min(self.cap_max - self.storage, 200) - eps,
                                        name="storage_in_hat")
            storage_out_hat = mod.addVar(vtype=GRB.SEMICONT, lb=eps, ub=min(self.storage, 200) - eps,
                                         name="storage_out_hat")

            storage_in, storage_out, grid_in, diesel_power = action
            # power balance constraint and objective function
            pwr_bal = (grid_out == (
                    tilde_cons - tilde_ren - storage_out_hat - diesel_power_hat + storage_in_hat + grid_in_hat))
            obf = (((storage_in - storage_in_hat) ** 2) + ((storage_out - storage_out_hat) ** 2) +
                   ((grid_in - grid_in_hat) ** 2) + ((diesel_power - diesel_power_hat) ** 2))
        else:  # toy variant
            grid_in, diesel_power = action
            # power balance constraint and objective function
            pwr_bal = (grid_out == (tilde_cons - tilde_ren - diesel_power_hat + grid_in_hat))
            obf = (((grid_in - grid_in_hat) ** 2) + ((diesel_power - diesel_power_hat) ** 2))

        mod.addConstr(pwr_bal, "power_balance")
        mod.addConstr(grid_out >= eps, "numerical stability")
        mod.setObjective(obf)

        # get the closest feasible action
        feasible = self.optimize(mod)
        assert feasible
        if isinstance(self, envs.StandardVPPEnv):
            closest = np.array([mod.getVarByName(var).X
                                for var in ('storage_in_hat', 'storage_out_hat', 'grid_in_hat', 'diesel_power_hat')],
                               dtype=np.float64)
        else:
            closest = np.array([mod.getVarByName(var).X for var in ('grid_in_hat', 'diesel_power_hat')],
                               dtype=np.float64)

        closest[np.abs(closest) < 1e-10] = 0  # numerical instability
        assert not mod.getVarByName('grid_out').X < 0
        return closest
