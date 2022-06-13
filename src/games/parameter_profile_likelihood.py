#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:24:39 2022

@author: kate
"""
import pandas as pd
import numpy as np
from math import sqrt
from set_model import model
import matplotlib.pyplot as plt
from optimization import Optimization
from config import Settings, ExperimentalData
from solve_single import solve_single_parameter_set
from analysis import calc_chi_sq

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
    print('REFERENCE PARAMETERS - ORIGINAL EXPERIMENTAL DATA')  
    model.parameters = p_ref
    norm_solutions_ref, chi_sq_ref, r_sq_ref = solve_single_parameter_set()
    print('chi_sq REFERENCE: ' + str(round(chi_sq_ref, 4)))
    print('******************')
    
    #Generate noise realizations and calculate chi_sq_ref for each noise realization
    n_noise_realizations = 10
    exp_data_original = ExperimentalData.exp_data
    exp_data_noise_list = []
    chi_sq_ref_list = []

    #Define mean and standard error for error distribution
    mu = 0
    sigma =  .05 / sqrt(3)
    
    #Generate noise array
    np.random.seed(6754)
    noise_array = np.random.normal(mu, sigma, (n_noise_realizations, len(exp_data_original)))

    for i in range(0, n_noise_realizations):
        #add noise and append to experimental data to list for saving
        exp_data_noise = []
        for j, val in enumerate(exp_data_original):
            new_val = val + noise_array[i, j]
            if new_val < 0.0:
                new_val = 0.0
            exp_data_noise.append(new_val)
   
        # #Re-normalize the data to the maximum value in the dataset
        # if data == 'ligand dose response and DBD dose response':
        #     data_ligand = exp_data_noise[:11]
        #     data_DBD = exp_data_noise[11:]
            
        #     L_norm = [i/max(data_ligand) for i in data_ligand]
        #     DBD_norm = [i/max(data_DBD) for i in data_DBD]
        #     exp_data_noise_ = L_norm + DBD_norm
    
        # else:
        exp_data_noise_norm = [i/max(exp_data_noise) for i in exp_data_noise]
        exp_data_noise_list.append(exp_data_noise_norm)
        
        #Calculate chi_sq_ref using noise realization i
        chi_sq = calc_chi_sq(norm_solutions_ref, exp_data_noise_norm, ExperimentalData.exp_error)
        chi_sq_ref_list.append(chi_sq_ref)
      
    print('Calculating chi_sq FIT...')   

    #Run optimization on each noise realization with the calibrated parameters as the initial guess
    df_noise = pd.DataFrame()
    for i, parameter_label in enumerate(Settings.parameter_labels):
        df_noise[parameter_label] = [calibrated_parameters[i]] * n_noise_realizations
    df_noise['data'] = exp_data_noise_list
    _, _, df_optimization_results, _ = Optimization.optimize_all(df_noise, 'PPL')
    chi_sq_fit_list = list(df_optimization_results['chi_sq'])

    #Plot chi_sq distribution and calculate 99% CI threshold
    chi_sq_distribution = []
    for i in range(0, len(chi_sq_ref_list)):
        chi_sq_distribution.append(chi_sq_ref_list[i] - chi_sq_fit_list[i])
     
    #Define threshold value
    confidence_interval_label = Settings.confidence_interval*100
    threshold_chi_sq= np.percentile(chi_sq_distribution, confidence_interval_label)
  
    #Plot results
    plt.figure(figsize = (5,3))
    plt.xlabel('chi_sq_ref - chi_sq_fit')
    plt.ylabel('Count')
    y, x, _ = plt.hist(chi_sq_distribution, bins = 35, histtype='bar', color = 'dimgrey')
    plt.plot([threshold_chi_sq, threshold_chi_sq], [0, max(y)], '-', lw=3, 
             alpha=0.6, color = 'dodgerblue', linestyle = ':', 
             label = str(confidence_interval_label) + '%')  
    plt.savefig('./CHI_SQ DISTRIBUTION', bbox_inches="tight", dpi = 600)
    
    print('******************')
    threshold_chi_sq= np.round(threshold_chi_sq, 1)
    print('chi_sq THRESHOLD FOR ' + str(len(Settings.parameters)) + ' PARAMETERS WITH ' + 
          str(confidence_interval_label) + '% CONFIDENCE')
    print(threshold_chi_sq)
    print('******************')
    
    #Save results
    filename = 'CONDITIONS PL' 
    with open(filename + '.txt', 'w') as file:
        file.write('threshold_chi_sq: ' + str(threshold_chi_sq) + '\n')
        file.write('\n')
        file.write('parameter_labels: ' + str(Settings.parameter_labels) + '\n')
        file.write('\n')
        file.write('calibrated_params: ' + str(calibrated_parameters) + '\n')
        file.write('\n')
        file.write('calibrated_chi_sq: ' + str(calibrated_chi_sq) + '\n')
    print('Conditions saved.')

    return threshold_chi_sq

def calculate_profile_likelihood(parameter_label: str, calibrated_parameters: list, calibrated_chi_sq: float) -> pd.DataFrame:
    """Calculates the PPL for the given parameter
    
    Parameters
    ----------
    parameter_label
        a string defining the parameter label 
        
    calibrated_parameters
        a list of floats containing the calibrated values for each parameter
        
    calibrated_chi_sq
        a float defining the chi_sq associated with the calibrated parameter set

    Returns
    -------
    df_profile_likelihood_results
        a df containing the PPL results
    
    """
    pass
    
    
    
    #return df_profile_likelihood_results


def plot_parameter_profile_likelihood_consequences(df_profile_likelihood_results):
    pass
    