#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:02:58 2022

@author: kate
"""

from math import sqrt
import pandas as pd
import numpy as np
from models.set_model import model
from modules.parameter_estimation.optimization import optimize_all
from config.settings import settings
from config.experimental_data import ExperimentalData
from modules.solve_single import solve_single_parameter_set
from utilities.metrics import calc_chi_sq
from utilities.saving import save_chi_sq_distribution
from plots.plots_parameter_profile_likelihood import plot_chi_sq_distribution

def generate_noise_realizations_and_calc_chi_sq_ref(norm_solutions_ref: list):
    """Generates noise realizations and calculates chi_sq_ref
     
    Parameters
    ----------
    norm_solutions_ref
        a list of floats containing the solutions generated with the reference parameters

    Returns
    -------
    exp_data_noise_list
        a list of lists containing the noise realizations 
        (length = # noise realizations)
    
    chi_sq_ref_list
        a list of floats containing the chi_sq_ref values for each noise realization 
        (length = # noise realizations)

    """
    # Generate noise realizations and calculate chi_sq_ref for each noise realization
    exp_data_original = ExperimentalData.exp_data
    exp_data_noise_list = []
    chi_sq_ref_list = []

    # Define mean and standard error for error distribution
    mu = 0
    sigma = 0.05 / sqrt(3)

    # Generate noise array
    np.random.seed(6754)
    noise_array = np.random.normal(mu, sigma, (settings["num_noise_realizations"], len(exp_data_original)))
    
    for i in range(0, settings["num_noise_realizations"]):
        # add noise and append to experimental data to list for saving
        exp_data_noise = []
        for j, val in enumerate(exp_data_original):
            new_val = val + noise_array[i, j]
            new_val = max(new_val, 0.0)
            exp_data_noise.append(new_val)
            
        #re-normalize data
        exp_data_noise_norm = [i / max(exp_data_noise) for i in exp_data_noise]
        exp_data_noise_list.append(exp_data_noise_norm)

        # Calculate chi_sq_ref using noise realization i
        chi_sq = calc_chi_sq(norm_solutions_ref, exp_data_noise_norm, ExperimentalData.exp_error)
        chi_sq_ref_list.append(chi_sq)
        
    return exp_data_noise_list, chi_sq_ref_list

def calculate_chi_sq_fit(calibrated_parameters: list, exp_data_noise_list: list) -> list:
    """Runs optimization on each noise realization with the calibrated parameters as the initial guess

    Parameters
    ----------  
    calibrated_parameters
        a list of floats defining the calibrated parmaeter values
        
    exp_data_noise_list
        a list of lists containing the noise realizations 
        (length = # noise realizations)
        
    Returns
    -------
    chi_sq_fit_list
        a list of floats defining the chi_sq_fit values for each noise realization

    """
    
    df_noise = pd.DataFrame()
    for i, parameter_label in enumerate(settings["parameter_labels"]):
        df_noise[parameter_label] = [calibrated_parameters[i]] * settings["num_noise_realizations"]
    df_noise["data"] = exp_data_noise_list
    _, _, df_optimization_results, _ = Optimization.optimize_all(df_noise, "PPL")
    chi_sq_fit_list = list(df_optimization_results["chi_sq"])

    return chi_sq_fit_list

def calculate_threshold_chi_sq(calibrated_parameters: list, calibrated_chi_sq: float) -> float:
    """Calculates threshold chi_sq value for PPL calculations

    Parameters
    ----------
    calibrated_parameters
        a list of floats containing the calibrated values for each parameter

    calibrated_chi_sq
        a float defining the chi_sq associated with the calibrated parameter set

    Returns
    -------
    threshold_chi_sq
        a float defining the threshold chi_sq value

    """

    p_ref = [15, 1, 0.05, 720, 100, 2]
    print("REFERENCE PARAMETERS - ORIGINAL EXPERIMENTAL DATA")
    model.parameters = p_ref
    norm_solutions_ref, chi_sq_ref, _ = solve_single_parameter_set()
    print("chi_sq REFERENCE: " + str(round(chi_sq_ref, 4)))
    print("******************")
    
    print("Generating noise realizations and calculating chi_sq_ref...")
    exp_data_noise_list, chi_sq_ref_list = generate_noise_realizations_and_calc_chi_sq_ref(norm_solutions_ref)
    
    print("Calculating chi_sq_fit...")
    chi_sq_fit_list = calculate_chi_sq_fit(calibrated_parameters, exp_data_noise_list)
    
    # Calculate threshold chi_sq
    chi_sq_distribution = []
    for i, ref_val in enumerate(chi_sq_ref_list):
        chi_sq_distribution.append(ref_val - chi_sq_fit_list[i])
    confidence_interval = 99
    threshold_chi_sq = np.percentile(chi_sq_distribution, confidence_interval)
     
    plot_chi_sq_distribution() 
    save_chi_sq_distribution() 
    
    print("******************")
    threshold_chi_sq_rounded = np.round(threshold_chi_sq, 1)
    print(
        "chi_sq threshold for "
        + str(len(settings["parameters"]))
        + " parameters with "
        + str(confidence_interval)
        + "% confidence"
    )
    print(threshold_chi_sq_rounded)
    print("******************")

    return threshold_chi_sq
