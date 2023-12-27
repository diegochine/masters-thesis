import numpy as np
import pandas as pd
from gurobipy import Model, GRB
from tqdm import trange

from src.envs import VPPEnv


def optimal_solver(c_grid: np.ndarray,
                   shifts: np.ndarray,
                   p_ren_pv_real: np.ndarray,
                   tot_cons_real: np.ndarray) -> float:
    """
    This is the implementation of the optimal solver. Basically, rather than considering one stage at a time it assumes
    to know the future realizations and solve all the stages in a single problem formulation.
    :param c_grid: np.ndarray; electricity prices from the market.
    :param shifts: np.ndarray; precomputed optimal shifts
    :param p_ren_pv_real: np.ndarray; realizations of the photovoltaic production.
    :param tot_cons_real: np.ndarray; realizations of the user demands.
    :return: float; optimal cost.
    """

    # Number of timestamp
    n = 96

    # Capacities, bounds, parameters and prices
    cap_max = 1000
    in_cap = 500
    c_diesel = 0.054
    p_diesel_max = 1200

    # Initialize the storage capacitance
    cap_x = in_cap

    # FIXME: legacy code, not very elegant initialization
    cap, p_diesel, p_storage_in, p_storage_out, p_grid_in, p_grid_out, tilde_cons = [[None] * n for _ in range(7)]

    obj = 0

    # Create a Gurobi model
    mod = Model()

    # Loop for each timestep
    for i in range(n):

        # Build variables and define bounds
        p_diesel[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="p_diesel_" + str(i))
        p_storage_in[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="p_storage_in_" + str(i))
        p_storage_out[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="p_storage_out_" + str(i))
        p_grid_in[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="p_grid_in_" + str(i))
        p_grid_out[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="p_grid_out_" + str(i))
        cap[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="cap_" + str(i))

        # Shifts from Demand Side Energy Management System
        tilde_cons[i] = (shifts[i] + tot_cons_real[i])

        # Power balance constraint
        mod.addConstr((p_ren_pv_real[i] + p_storage_out[i] + p_grid_out[i] + p_diesel[i] -
                       p_storage_in[i] - p_grid_in[i] == tilde_cons[i]), "Power_balance")

        # Storage capacity constraint
        mod.addConstr(cap[i] == cap_x + p_storage_in[i] - p_storage_out[i])
        mod.addConstr(cap[i] <= cap_max)

        mod.addConstr(p_storage_in[i] <= cap_max - cap_x)
        mod.addConstr(p_storage_out[i] <= cap_x)

        mod.addConstr(p_storage_in[i] <= 200)
        mod.addConstr(p_storage_out[i] <= 200)

        # Diesel and grid bounds
        mod.addConstr(p_diesel[i] <= p_diesel_max)
        mod.addConstr(p_grid_in[i] <= 600)

        # Objective function
        obf = (c_grid[i] * p_grid_out[i] + c_diesel * p_diesel[i] - c_grid[i] * p_grid_in[i])

        obj += obf

        # Update the storage capacity
        cap_x = cap[i]

    mod.setObjective(obj)
    mod.optimize()

    # Update the optimal cost
    optimal_cost = mod.objVal

    return optimal_cost


if __name__ == '__main__':
    # execute with python -m src.utils.compute_optims
    instances = pd.read_csv('src/data/Dataset10k.csv', index_col=0)
    instances = VPPEnv.instances_preprocessing(instances)
    c_grid = np.load('src/data/gmePrices.npy')
    shifts = np.load('src/data/optShift.npy')
    noise_std_dev = 0.02

    for instance in trange(10000):
        # These are the forecasted photovoltaic production and user demands
        p_ren_pv_forecasted = instances.iloc[instance]['PV(kW)']
        tot_cons_forecasted = instances.iloc[instance]['Load(kW)']
        p_ren_pv_forecasted = np.asarray(p_ren_pv_forecasted)
        tot_cons_forecasted = np.asarray(tot_cons_forecasted)

        # Here we create a specific realization from the forecasts
        # noise is assumed to be generated with numpy default RNG seeded with the instance number
        rng = np.random.default_rng(instance)
        noise_pv = rng.normal(0, noise_std_dev, 96)
        noise_load = rng.normal(0, noise_std_dev, 96)

        p_ren_pv_real = p_ren_pv_forecasted + p_ren_pv_forecasted * noise_pv
        tot_cons_real = tot_cons_forecasted + tot_cons_forecasted * noise_load

        opt_cost = optimal_solver(c_grid=c_grid, shifts=shifts, tot_cons_real=tot_cons_real, p_ren_pv_real=p_ren_pv_real)
        np.save(f'src/data/oracle/{instance}_cost.npy', opt_cost)
