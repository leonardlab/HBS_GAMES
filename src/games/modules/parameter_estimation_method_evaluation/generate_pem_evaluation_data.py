#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:13:15 2022

@author: kate
"""
import json
from math import sqrt
import pandas as pd
import numpy as np
from games.models.set_model import model
from games.modules.solve_single import solve_single_parameter_set
from games.utilities.metrics import calc_chi_sq, calc_r_sq
from games.config.settings import settings
from games.utilities.saving import save_pem_evaluation_data


def add_noise(solutions_norm_raw: list, count: int) -> list:
    """Adds noise to a set of simulated data

    Parameters
    ----------
    solutions_norm_raw
        a list of floats containing the raw simulation results

    count
        an integer defining the PEM evaluation dataset number

    Returns
    -------
    solutions_norm_noise
        a list of flaots containing the simulation results with added noise
    """
    # Define mean and std for error distribution
    mu = 0
    sigma = 0.05 / sqrt(3)

    # Generate noise values
    seeds = [3457, 1234, 2456, 7984, 7306, 3869, 5760, 9057, 2859]
    np.random.seed(seeds[count])
    noise = np.random.normal(mu, sigma, len(solutions_norm_raw))

    # Add noise to each data point
    solutions_noise = []
    for i, noise_ in enumerate(noise):
        new_val = solutions_norm_raw[i] + noise_
        new_val = max(new_val, 0.0)
        solutions_noise.append(new_val)

    solutions_norm_noise = [i / max(solutions_noise) for i in solutions_noise]

    return solutions_norm_noise


def generate_pem_evaluation_data(df_global_search_results: pd.DataFrame) -> list:
    """Generates PEM evaluation data based on results of a global search

    Parameters
    ----------
    df_global_search
        a dataframe containing global search results

    Returns
    -------
    pem_evaluation_data_list
        a list of lists containing the PEM evaluation data

    Files
    ----------
    "PEM evaluation criterion.json"
        contains PEM evaluation criterion using both chi_sq and r_sq

    """

    # filter data to choose parameter sets used to generate PEM evaluation data
    df_global_search_results = df_global_search_results.sort_values(by=["chi_sq"])
    df_global_search_results = df_global_search_results.reset_index(drop=True)
    df_global_search_results = df_global_search_results.drop(
        df_global_search_results.index[settings["num_pem_evaluation_datasets"] :]
    )

    # Define, add noise to, and save PEM evaluation data
    count = 1
    pem_evaluation_data_list = []
    r_sq_list = []
    chi_sq_list = []
    for row in df_global_search_results.itertuples(name=None):
        # Define parameters
        p = list(row[1 : len(settings["parameters"]) + 1])

        # Solve for raw data
        model.parameters = p
        solutions_norm_raw, chi_sq, r_sq = solve_single_parameter_set()

        # Add noise
        solutions_norm_noise = add_noise(solutions_norm_raw, count)

        # Calculate cost function metrics between PEM evaluation
        # training data with and without noise
        r_sq = calc_r_sq(solutions_norm_raw, solutions_norm_noise)
        r_sq_list.append(r_sq)
        chi_sq = calc_chi_sq(solutions_norm_raw, solutions_norm_noise)
        chi_sq_list.append(chi_sq)

        pem_evaluation_data_list.append(solutions_norm_noise)
        count += 1

    save_pem_evaluation_data(pem_evaluation_data_list)

    # Define PEM evaluation criterion
    mean_r_sq = np.round(np.mean(r_sq_list), 4)
    min_r_sq = np.round(min(r_sq_list), 4)
    print("Mean R2 between PEM evaluation data with and without noise: " + str(mean_r_sq))
    print("Min R2 between PEM evaluation data with and without noise: " + str(min_r_sq))

    mean_chi_sq = np.round(np.mean(chi_sq_list), 4)
    max_chi_sq = np.round(max(chi_sq_list), 4)
    print("Mean chi_sq between PEM evaluation data with and without noise: " + str(mean_chi_sq))
    print("Max chi_sq between PEM evaluation data with and without noise: " + str(max_chi_sq))

    # Save PEM evaluation criterion
    with open("PEM evaluation criterion.json", "w", encoding="utf-8") as file:
        json.dump(r_sq_list, file, indent=2)
        json.dump(chi_sq_list, file, indent=2)

    return pem_evaluation_data_list, max_chi_sq
