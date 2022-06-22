#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 15:25:19 2022

@author: kate
"""
from typing import Tuple
import multiprocessing as mp
import numpy as np
import pandas as pd
from SALib.sample import latin
from games.models.set_model import model
from games.modules.solve_single import solve_single_parameter_set
from games.config.settings import settings


def generate_parameter_sets(problem_global_search: dict, all_parameters: list = settings["parameters"]) -> pd.DataFrame:
    """
    Generate parameter sets for global search

    Parameters
    ----------
    problem_global_search
        a dictionary including the number, labels, and bounds for the free parameters

    fixed_parameters
        a list of floats containing all initial parameter values,
        including fixed and free parameters


    Returns
    -------
    df_parameters
        df with columns defining to parameter identities
        and rows defining the paramter values for each set in the sweep
    """

    fit_params = problem_global_search[
        "names"
    ]  # only the currently free parameters (as set in settings)
    num_params = problem_global_search["num_vars"]
    all_params = settings["parameter_labels"]  # all params that are potentially free
    n_search = settings["num_parameter_sets_global_search"]

    # Create an empty dataframe to store results of the parameter sweep
    df_parameters = pd.DataFrame()

    # Fill each column of the dataframe with the intial values set in settings["
    for i, _ in enumerate(all_params):
        param = all_params[i]
        param_array = np.full((1, n_search), all_parameters[i])
        param_array = param_array.tolist()
        df_parameters[param] = param_array[0]

    # Perform LHS
    param_values = latin.sample(problem_global_search, n_search, seed=456767)

    # Transform back to to linear scale
    params_converted = []
    for item in param_values:
        params_converted.append([10 ** (val) for val in item])
    params_converted = np.asarray(params_converted)

    # Replace the column of each fit parameter with the list of parameters from the sweep
    for item in range(0, num_params):
        for name in fit_params:
            if fit_params[item] == name:
                df_parameters[name] = params_converted[:, item]

    df_parameters.to_csv("parameter sweep.csv")

    return df_parameters


def solve_single_for_global_search(row: tuple) -> Tuple[list, float]:
    """
    Solves equation for a single parameter set - structure is
    necessary for upstream multiprocessing code

    Parameters
    ----------
    row
        tuple defining the parameters and experimental data

    Returns
    -------
    solutions
        a list of floats defining the simulation solutions

    chi_sq
        a float defining the chi_sq value

    """
    #Unpack row
    model.parameters = list(row[1:-3])
    x = row[-3]
    exp_data = row[-2]
    exp_error = row[-1]

    #Solve equations
    solutions, chi_sq, _ = solve_single_parameter_set(x, exp_data, exp_error)
    return solutions, chi_sq


def solve_global_search(df_parameters: pd.DataFrame, x: list, exp_data: list, exp_error: list, run_type: str = "default") -> pd.DataFrame:
    """
    Generates parameter sets for global search

    Parameters
    ----------
    df_parameters
        df with columns defining to parameter identities
        and rows defining the parameter values for each set in the sweep
    x
        a list of floats containing the values of the independent variable

    exp_data
        a list of floats containing the values of the dependent variable

    exp_error
        a list of floats containing the values of the measurement error
        for the dependent variable

    run_type
        a string defining the run_type ('default' or 'PEM evaluation')

    Returns
    -------
    df_global_search_results
        df that contains the information in df_parameters,
        along with an extra column defining the cost function
        for each parameter set

    """
    #Add experimental data items to df
    df_parameters['x'] = [x] * len(df_parameters.index)
    df_parameters['exp_data'] = [exp_data] * len(df_parameters.index)
    df_parameters['exp_error'] = [exp_error] * len(df_parameters.index)

    #Solve for each parameter set in global search
    chi_sq_list = []
    solutions_list = []
    if settings["parallelization"] == "no":
        for row in df_parameters.itertuples(name=None):
            solutions, chi_sq = solve_single_for_global_search(row)
            solutions_list.append(solutions)
            chi_sq_list.append(chi_sq)

    elif settings["parallelization"] == "yes":
        with mp.Pool(settings["num_cores"]) as pool:
            result = pool.imap(solve_single_for_global_search, df_parameters.itertuples(name=None))
            pool.close()
            pool.join()
            output = [[list(x[0]), round(x[1], 4)] for x in result]

        for i, item in enumerate(output):
            solutions_list.append(item[0])
            chi_sq_list.append(item[1])

    #structure results
    df_global_search_results = df_parameters
    df_global_search_results["chi_sq"] = chi_sq_list
    df_global_search_results["normalized solutions"] = solutions_list

    df_global_search_results.to_csv("global search results.csv")
    return df_global_search_results
