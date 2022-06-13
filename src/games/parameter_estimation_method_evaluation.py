#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:49:45 2022

@author: kate
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from math import sqrt
from openpyxl import load_workbook
from config import Settings, Context, ExperimentalData, create_folder
from set_model import model
from solve_single import solve_single_parameter_set
from analysis import calc_chi_sq, calc_r_sq
from optimization import Optimization

def plot_pem_evaluation(df_list: list, chi_sq_pem_evaluation_criterion: float) -> None:
    """Plots results of PEM evaluation runs

    Parameters
    ----------
    df_list
        a list of dataframes defining the optimization results (length = #PEM evaluation data sets)
        
    chi_sq_pem_evaluation_criterion
        a float defining the pem evaluation criterion for chi_sq

    Returns
    -------
    None

    Figures:
    -------
    'PEM EVALUATION CRITERION ALL OPT.svg' 
        (plot of ALL optimized chi_sq values for each PEM evaluation dataset)
        
    'PEM EVALUATION CRITERION.svg' 
        (plot of the best optimized chi_sq values for each PEM evaluation dataset)
    """
    run = []
    chi_sq_list = []
    r_sq_list = []
    min_cf_list = []
    max_r_sq_list = []
    min_cf_list = []
    min_cf_list = []
  
    for i, df in enumerate(df_list):
        chi_sq = list(df['chi_sq'])
        r_sq = list(df['r_sq'])
        chi_sq_list = chi_sq_list + chi_sq
        r_sq_list = r_sq_list + r_sq
        run_ = [i + 1] * len(r_sq)
        run = run + run_
        min_cf_list.append(min(chi_sq))
     
    df_all = pd.DataFrame(columns = ['run', 'chisq', 'r_sq'])
    df_all['run'] = run
    df_all['chi_sq'] = chi_sq_list
    df_all['r_sq'] = r_sq_list
    
    #Plot PEM evaluation criterion
    plt.subplots(1,1, figsize=(4,3))
    ax1 = sns.boxplot(x='run', y='chi_sq', data=df_all, color = 'white')
    ax1 = sns.swarmplot(x='run', y='chi_sq', data=df_all, color="black")
    ax1.set(xlabel='PEM evaluation dataset', ylabel='chi_sq, opt', title = 'chi_sq_pass = ' + str(chi_sq_pem_evaluation_criterion))
    plt.savefig('PEM EVALUATION CRITERION ALL OPT.svg', dpi = 600)
   
    plt.subplots(1,1, figsize=(4,3))
    plt.plot(range(1, 4), min_cf_list, color = 'black', marker = 'o', linestyle = 'None')
    plt.xlabel('PEM evaluation dataset')
    plt.ylabel('chi_sq, min')
    plt.plot([1, 3], [chi_sq_pem_evaluation_criterion, chi_sq_pem_evaluation_criterion], color = 'black', marker = 'None', linestyle = 'dotted')
    plt.xticks([1, 2, 3])
    plt.savefig('PEM EVALUATION CRITERION.svg', dpi = 600)
   
    

def define_initial_guesses_for_pem_eval(df_global_search_results: pd.DataFrame, pem_evaluation_data_list: list) -> list:
    """Defines initial guesses for each pem evaluation optimization run based on results of global search 
    PEM evaluation data and then running multi-start optimziation with each set of PEM evaluation data 

    Parameters
    ----------
    df_global_search_results
        a df containing global search results
        
    pem_evaluation_data_list
        a list of lists defining the pem evaluation data (length = # pem evaluation datasets)

    Returns
    -------
    df_initial_guesses_list
        a list of dataframes containing the initial guesses for each pem evaluation dataset
        (length = # pem evaluation datasets)

    """
    
    df_initial_guesses_list = []
    for i, pem_evaluation_data in enumerate(pem_evaluation_data_list):
        chi_sq_list = []
        for norm_solutions in list(df_global_search_results['normalized solutions']):
            chi_sq = calc_chi_sq(norm_solutions, pem_evaluation_data)
            chi_sq_list.append(chi_sq)
        label = 'chi_sq_' + str(i+1)
        df_global_search_results[label] = chi_sq_list
        df_ = df_global_search_results.sort_values(by=[label])
        df_ = df_.reset_index(drop = True)
        df_ = df_.drop(df_.index[0])
        df_ = df_.reset_index(drop = True)
        df_ = df_.drop(df_.index[Settings.num_parameter_sets_optimization:])
        df_initial_guesses_list.append(df_)

    return df_initial_guesses_list

def optimize_pem_evaluation_data(df_initial_guesses_list: list, pem_evaluation_data_list: list, chi_sq_pem_evaluation_criterion: float) -> None:
    """Runs optimization for each set of pem evaluation data

    Parameters
    ----------
    df_initial_guesses_list
        a list of dfs containing the initial guesses for each pem evaluation data set
        
    pem_evaluation_data_list
        a list of lists containing the pem evaluation data
        
    chi_sq_pem_evaluation_criterion
        a float defining the pem evaluation criterion for chi_sq
        
    Returns
    -------
    df_list
        a list of dataframes defining the optimization results (length = #PEM evaluation data sets)

    """
    r_sq_pem_evaluation = []
    chi_sq_pem_evaluation = []
    df_list = []
    
    for i, df in enumerate(df_initial_guesses_list):
        sub_folder_name = 'PEM evaluation data ' + str(i+1)
        path = create_folder('./MODULE 1 - EVALUATE PARAMETER ESTIMATION METHOD/' + sub_folder_name)
        os.chdir(path)
        
        ExperimentalData.exp_data = pem_evaluation_data_list[i]
        
        with pd.ExcelWriter("./INITIAL GUESSES.xlsx") as writer:
            df.to_excel(writer, sheet_name="")
            
        r_sq_mean, chi_sq_mean, df_optimization_results = Optimization.optimize_all(df)
        df_list.append(df_optimization_results)
        r_sq_pem_evaluation.append(r_sq_mean)
        chi_sq_pem_evaluation.append(chi_sq_mean)
        
        os.chdir(Context.folder_path + '/' + "MODULE 1 - EVALUATE PARAMETER ESTIMATION METHOD")
        
    r_sq_min = min(r_sq_pem_evaluation)
    chi_sq_max = max(chi_sq_pem_evaluation)
    
    print('chi_sq PEM evaluation criterion = ' + str(np.round(chi_sq_pem_evaluation_criterion, 4)))
    print('chi_sq max across all PEM evaluation = ' + str(np.round(chi_sq_max, 4)))
    
    print('r_sq min across all PEM evaluation = ' + str(np.round(r_sq_min, 4)))
    
    if chi_sq_max <= chi_sq_pem_evaluation_criterion:
        print('MODULE 1 PASSED')
    else:
        print('MODULE 1 FAILED')
        
    return df_list
        

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
    #Define mean and std for error distribution
    mu = 0
    sigma =  .05 / sqrt(3)
    
    #Generate noise values
    seeds = [3457, 1234, 2456, 7984, 7306, 3869, 5760, 9057, 2859]
    np.random.seed(seeds[count])
    noise = np.random.normal(mu, sigma, len(solutions_norm_raw))

    #Add noise to each data point
    solutions_noise = []
    for i, noise_ in enumerate(noise):
        new_val = solutions_norm_raw[i] + noise_
        if new_val < 0.0:
            new_val = 0.0
        solutions_noise.append(new_val)
    
    solutions_norm_noise = [i/max(solutions_noise) for i in solutions_noise]
        
    return solutions_norm_noise
    
def save_pem_evaluation_data(solutions_norm_noise: list, filename: str, count: int):
    """
    Saves PEM evaluation data in format usable by downstream code
    
    Parameters
    ----------
    solutions_norm_raw
        a list of floats containing the raw simulation results
        
    filename
        a string defining the filename to save the results to
        
    count
        an integer defining the data set number that is being saved 

    Returns
    -------
    df_test
         a dataframe containing the PEM evaluation data
    """
 
    #Define the dataframe
    if Settings.dataID == 'ligand dose response':
        df_test1 = pd.DataFrame(solutions_norm_noise, columns = ['L'])
        df_test = pd.concat([df_test1], axis=1)
       
    #Save results
    filename = filename + '.xlsx'
    if count==1:
        with pd.ExcelWriter(filename) as writer:  # doctest: +SKIP
            df_test.to_excel(writer, sheet_name = str(count))
    else: 
        path = filename
        book = load_workbook(path)
        writer = pd.ExcelWriter(path, engine = 'openpyxl')
        writer.book = book      
        df_test.to_excel(writer, sheet_name = str(count))
        writer.save()
        writer.close() 
    return df_test
   
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
 
    #filter data to choose parameter sets used to generate PEM evaluation data
    df_global_search_results = df_global_search_results.sort_values(by=["chi_sq"])
    df_global_search_results = df_global_search_results.reset_index(drop = True)
    df_global_search_results = df_global_search_results.drop(df_global_search_results.index[Settings.num_pem_evaluation_datasets:])
        
    #Define, add noise to, and save PEM evaluation data
    count = 1
    pem_evaluation_data_list = []
    r_sq_list = []
    chi_sq_list = []
    for row in df_global_search_results.itertuples(name = None):
        #Define parameters
        p = list(row[1 : len(Settings.parameters) + 1])
      
        #Solve for raw data
        model.parameters = p
        solutions_norm_raw, chi_sq, r_sq = solve_single_parameter_set()
    
        #Add noise
        solutions_norm_noise = add_noise(solutions_norm_raw, count)
        
        #Calculate cost function metrics between PEM evaluation training data with and without noise
        r_sq = calc_r_sq(solutions_norm_raw, solutions_norm_noise)
        r_sq_list.append(r_sq)
        chi_sq = calc_chi_sq(solutions_norm_raw, solutions_norm_noise)
        chi_sq_list.append(chi_sq)
   
        #Save results
        save_pem_evaluation_data(solutions_norm_raw,'PEM EVALUATION DATA RAW', count)
        df_noise = save_pem_evaluation_data(solutions_norm_noise,'PEM EVALUATION DATA NOISE', count)
        pem_evaluation_data_list.append(solutions_norm_noise)
        count += 1
    
    #Define PEM evaluation criterion
    mean_r_sq = np.round(np.mean(r_sq_list), 4)
    min_r_sq = np.round(min(r_sq_list), 4)
    print('Mean R2 between PEM evaluation data with and without noise: ' + str(mean_r_sq))
    print('Min R2 between PEM evaluation data with and without noise: ' + str(min_r_sq))
    
    mean_chi_sq = np.round(np.mean(chi_sq_list), 4)
    max_chi_sq = np.round(max(chi_sq_list), 4)
    print('Mean chi_sq between PEM evaluation data with and without noise: ' + str(mean_chi_sq))
    print('Max chi_sq between PEM evaluation data with and without noise: ' + str(max_chi_sq))
    
    #Save PEM evaluation criterion
    with open("PEM evaluation criterion.json", 'w') as f:
        json.dump(r_sq_list, f, indent=2) 
        json.dump(chi_sq_list, f, indent=2) 
  
    return pem_evaluation_data_list, max_chi_sq