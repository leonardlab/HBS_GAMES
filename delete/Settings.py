#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:12:12 2020

@author: kate
"""
#Package imports
from math import log10
import pandas as pd

#GAMES imports
from Saving import makeMainDir
from DefineExpData import defineExp


def init():
    '''
    Purpose: Define conditions for simulations, called by other functions to import conditions
    
    Input: None
    
    Outputs: 3 dictionaries - conditions_dictionary, initial_params_dictionary, data_dictionary
    
    '''
    # =============================================================================
    # 1. Define and create folder for saving results
    # =============================================================================
    #This will be the name of the run-specific results folder. 
    folder_name = '2021-10-14 model D M1 1000 + 100'
    
    # =============================================================================
    # 2. Define free parameters and bounds
    # =============================================================================
    #Set list of all potentially free parameters
    #model A/B
    #p_ref = [15, 0.05, 0.05, 36, 100, 2] #all potentially free parameters - for models A and B 
    #p_all = p_ref

    #model C
    #p_ref = [15, 1, .05, 720, 100, 2] #for model C (m = m*)
    #p_all = p_ref
  
    #model D
    p_ref = [15, 1, 0.05, 720, 100, 2] #for model D (m = m*)
    p_all = [15, 1, 1, 720, 100, 2] #for model D (m = m*)
    
    #Define parameter labels (real and general)
    real_param_labels_all = ['e', 'b', 'k_bind', 'm', 'km', 'n'] #real labels for p_ref and p_all
    p_labels_all = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6'] #general param labels
    
    #if a param in real_param_labels_all is not included in realParamLabels_free,
    #it is fixed at the value set in p_all
    real_param_labels_free = ['e', 'm', 'km', 'n'] #real labels for free params

    #Change param labels to generalizable param labels
    num_free_params = len(real_param_labels_free)
    initial_params_dictionary = {}
    params = []
    p_ref_free = []
    p_labels_free = []
    for i, value in enumerate(p_all):
        label = real_param_labels_all[i]
        if label in real_param_labels_free:
            initial_params_dictionary[label] = value
            p_labels_free.append(p_labels_all[i])
            params.append(value)
            p_ref_free.append(p_ref[i])
    
    #Set bounds for parameter estimation (+/- 3 orders of magnitude from the ref parameter value)
    bounds_log = []
    for i in range(0, num_free_params):
        min_bound = log10(p_ref_free[i]) - 3
        max_bound = log10(p_ref_free[i]) + 3
        bounds_log.append([min_bound, max_bound])
  
    #n is an exception to this because it has a biologically 
    #relevant range from 1 -> 4 (in linear scale)
    bounds_log[-1] = [0, .602]
    
    #Define the parameter estimation problem (free parameters for this run only)
    problem = {'num_vars': num_free_params,  #set free parameters and bounds
               'names': p_labels_free, 
               'bounds': bounds_log} #bounds are in log scale
    # =============================================================================
    # 3. Define conditions dictionary
    # =============================================================================
    #Initialize conditions dictionary
    #Items that you might want to change
    conditions_dictionary = {}
    conditions_dictionary["model"] = 'model C'
    conditions_dictionary["modules"] = [1, 2, 3] #[1,2,3] or [1,2] or [2,3] or [1] or [2] or [3] or [] for test only
    conditions_dictionary["n_search"] = 1000
    conditions_dictionary["n_initial_guesses"] = 100
    conditions_dictionary["confidence_interval"] = .99 
    conditions_dictionary["num_cores"] = 8
    conditions_dictionary["num_datasets_pem_eval"] = 8
    conditions_dictionary["n_search_pem_eval"] = 1000
    conditions_dictionary["param_index_PL"] = 'all' #'all' or index of p (int)
    
    if conditions_dictionary["model"] == 'model A':
        conditions_dictionary["data"] = 'ligand dose response only'
    elif conditions_dictionary["model"] in ['model B', 'model C', 'model D']:
        conditions_dictionary["data"] = 'ligand dose response and DBD dose response'

    #Define parameter labels
    #Items that you likely will not change
    full_path = makeMainDir(folder_name)
    conditions_dictionary["real_param_labels_free"] = real_param_labels_free
    problem_all_params = {'num_vars': len(p_labels_all),  
                          'names': p_labels_all, 
                          'bounds': [[]]} 
    conditions_dictionary["real_param_labels_all"] = real_param_labels_all
    conditions_dictionary["p_all"] = p_all
    conditions_dictionary["p_ref_free"] = p_ref_free
    conditions_dictionary["p_labels_free"] = p_labels_free
    conditions_dictionary["p_ref"] = p_ref
    conditions_dictionary["directory"] = full_path
    conditions_dictionary["run"] = '1'
    conditions_dictionary["problem"] = problem
    conditions_dictionary["problem_all_params"] = problem_all_params
    
    # =============================================================================
    # 4. Define data dictionary
    # =============================================================================
    data_dictionary = {}
    df_ref = pd.read_excel('./REFERENCE TRAINING DATA.xlsx')   
    x_vals, exp_data, error = defineExp(conditions_dictionary["data"], df_ref)
    data_dictionary["x_vals"] = x_vals
    data_dictionary["exp_data"] = exp_data
    data_dictionary["error"] = error
    data_dictionary["data_type"] = ''
    
    return conditions_dictionary, initial_params_dictionary, data_dictionary
