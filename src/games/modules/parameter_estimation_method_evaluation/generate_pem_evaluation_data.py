#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:13:15 2022

@author: kate
"""
from typing import Tuple, List
import json
from math import sqrt
import pandas as pd
import numpy as np
from games.models.set_model import model
from games.modules.solve_single import solve_single_parameter_set
from games.utilities.metrics import calc_chi_sq, calc_r_sq
from games.config.settings import settings
from games.config.experimental_data import normalize_data_by_maximum_value
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
        a list of floats containing the simulation results with added noise
    """
    # Define mean and std for error distribution
    mu = 0
    sigma = 0.05 / sqrt(3)

    # Generate noise values
    seeds = [3457, 1234, 2456, 7984, 7306, 3869, 5760, 9057, 2859]
    if count > len(seeds):
        print("Error: must add more seeds to generate more than 8 PEM evaluation datasets")
    np.random.seed(seeds[count])
    noise = np.random.normal(mu, sigma, len(solutions_norm_raw))

    # Add noise to each data point
    solutions_noise = []
    for i, noise_val in enumerate(noise):
        new_val = solutions_norm_raw[i] + noise_val
        new_val = max(new_val, 0.0)
        solutions_noise.append(new_val)

    solutions_norm_noise, _ = normalize_data_by_maximum_value(solutions_noise, settings["dataID"])

    return solutions_norm_noise


def filter_global_search_results(
    df_global_search_results: pd.DataFrame, num_pem_evaluation_datasets: int
) -> pd.DataFrame:
    """Filters data to choose parameter sets used to generate PEM evaluation data

    Parameters
    ----------
    df_global_search_results
        a dataframe containing global search results

    num_pem_evaluation_datasets
        an int defining the number of pem evaluation datasets to generate

    Returns
    -------
    df_global_search_results_filtered
        a dataframe containing the filtered global search results

    """
    df_global_search_results_filtered = df_global_search_results.sort_values(by=["chi_sq"])
    df_global_search_results_filtered = df_global_search_results_filtered.reset_index(drop=True)
    df_global_search_results_filtered = df_global_search_results_filtered.drop(
        df_global_search_results_filtered.index[num_pem_evaluation_datasets:]
    )

    return df_global_search_results_filtered


def generate_pem_evaluation_data(df_global_search_results: pd.DataFrame) -> Tuple[list, float]:
    """Generates PEM evaluation data based on results of a global search

    Parameters
    ----------
    df_global_search_results
        a dataframe containing global search results

    Returns
    -------
    pem_evaluation_data_list
        a list of lists containing the PEM evaluation data

    """

    df_global_search_results_filtered = filter_global_search_results(
        df_global_search_results, settings["num_pem_evaluation_datasets"]
    )

    # Define, add noise to, and save PEM evaluation data
    count = 1
    pem_evaluation_data_list: List[list] = []
    r_sq_list: List[float] = []
    chi_sq_list: List[float] = []
    for row in df_global_search_results_filtered.itertuples(name=None):
        # Define parameters
        p = list(row[1 : len(settings["parameters"]) + 1])
        model.parameters = p

        # Solve for raw data
        x = list(df_global_search_results["x"].iloc[0])
        exp_data = list(df_global_search_results["exp_data"].iloc[0])
        exp_error = list(df_global_search_results["exp_error"].iloc[0])
        solutions_norm_raw, chi_sq, r_sq = solve_single_parameter_set(x, exp_data, exp_error)

        # Add noise
        solutions_norm_noise = add_noise(solutions_norm_raw, count)

        # Calculate cost function metrics between PEM evaluation
        # training data with and without noise
        r_sq = calc_r_sq(solutions_norm_raw, solutions_norm_noise)
        chi_sq = calc_chi_sq(solutions_norm_raw, solutions_norm_noise, exp_error)

        # Add metrics to full lists
        r_sq_list.append(r_sq)
        chi_sq_list.append(chi_sq)
        pem_evaluation_data_list.append(solutions_norm_noise)
        count += 1

    max_chi_sq = define_pem_evaluation_criterion(r_sq_list, chi_sq_list)
    save_pem_evaluation_data(pem_evaluation_data_list)
    return pem_evaluation_data_list, max_chi_sq


def define_pem_evaluation_criterion(r_sq_list: list, chi_sq_list: list) -> float:
    """Generates PEM evaluation data based on results of a global search

    Parameters
    ----------
    r_sq_list
        a list of floats containing the r_sq values associated
        with each pem evaluation dataset
        (calculated between the data with and without noise)

    chi_sq_list
        a list of floats containing the chi_sq values associated
        with each pem evaluation dataset
        (calculated between the data with and without noise)

    Returns
    -------
    max_chi_sq
        a float defining the maximum chi_aq across all pem evaluation dataset
        (calculated between the data with and without noise)
        Note that this return can be changed if the user wants to use a different
        metric to define the PEM evaluation criterion

    """

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

    return max_chi_sq
