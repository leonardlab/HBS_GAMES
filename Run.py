#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 08:50:07 2020

@author: kate
"""

# =============================================================================
# IMPORTS
# =============================================================================
#Import external packages/functions
import datetime
import os
import multiprocessing as mp
import math
from math import log10
import warnings
import pandas as pd
import numpy as np
from lmfit import Model, Parameters
from openpyxl import load_workbook
import matplotlib.pyplot as plt 
import seaborn as sns
import cycler
from math import sqrt

#Import GAMES functions
from Solvers import solveSingle, calcRsq, calcChi2
from Saving import createFolder, saveConditions, savePL
from GlobalSearch import generateParams, filterGlobalSearch
import Settings
from Analysis import plotPemEvaluation

#ignore ODEint warnings that clog up the console - user can remove this line if they want to see the warnings
warnings.filterwarnings("ignore")

#Define colors for plotting (colors from a colorblind friendly palette)
black_ = [i/255 for i in [0, 0, 0]]
orange_ = [i/255 for i in [230, 159, 0]]
sky_blue = [i/255 for i in [86, 180, 233]]
pink_ = [i/255 for i in [204, 121, 167]]

#Unpack conditions from Settings.py
conditions_dictionary, initial_params_dictionary, data_dictionary = Settings.init()
run = conditions_dictionary["run"]
model = conditions_dictionary["model"] 
data = conditions_dictionary["data"]
num_cores = conditions_dictionary["num_cores"]
n_initial_guesses = conditions_dictionary["n_initial_guesses"]
problem_free = conditions_dictionary["problem"]
full_path = conditions_dictionary["directory"]
n_search = conditions_dictionary["n_search"]
n_search_parameter_estimation= conditions_dictionary["n_search"]
n_search_pem_eval = conditions_dictionary["n_search_pem_eval"]
modules = conditions_dictionary["modules"] 
confidence_interval = conditions_dictionary["confidence_interval"] 
fit_params = problem_free['names']
bounds = problem_free['bounds']
num_vars = problem_free['num_vars']
p_all = conditions_dictionary["p_all"] 
p_ref = conditions_dictionary["p_ref"] 
p_labels_free = conditions_dictionary["p_labels_free"] 
p_ref_free = conditions_dictionary["p_ref_free"] 
real_param_labels_all = conditions_dictionary["real_param_labels_all"] 
num_datasets_pem_eval = conditions_dictionary["num_datasets_pem_eval"] 
param_index_PL = conditions_dictionary["param_index_PL"]
problem_all_params = conditions_dictionary["problem_all_params"]
all_param_labels = problem_all_params['names']
param_labels = list(initial_params_dictionary.keys())
init_params = list(initial_params_dictionary.values())
real_param_labels_free = conditions_dictionary["real_param_labels_free"]
x = data_dictionary["x_vals"]
data_type = data_dictionary["data_type"]
exp_data = data_dictionary["exp_data"]
exp_data_original = data_dictionary["exp_data"]
error = data_dictionary["error"]
save_internal_states_flag = False
parallelization = 'yes' 

#Set style file
plt.style.use('./paper.mplstyle.py')

# =============================================================================
# General parameter estimation and solver code (used by modules 1, 2, 3)
# =============================================================================    
#Define doses
doses_ligand = [0] + list(np.logspace(0, 2, 10))
doses_dbd = [0, 2, 5, 10, 20, 50, 100, 200]

def solveAll(p, exp_data):
    
    '''
    Purpose: Solve ODEs for the entire dataset using parameters defined in p 
    
    Inputs: 
        p: a list of floats containing the parameter set (order of parameter defined in 
            Settings.py and in the ODE defintion function in Solvers.py)
            p = [e, b, k_bind, m, km, n]
        exp_data: a list defining the training data (# datapoints = length of list)
            
    Outputs: 
        if save_internal_states_flag == False:
            doses_: a list of floats corresponding to the x-axis values in the training data plots 
                (doses of ligand or plasmid doses of DBD)
            solutions: a list of floats corresponding to the y-axis values in the training data 
                plots (normalized reporter expression in this case)
            chi2: a float defining the chi2 for this parameter set
        if save_internal_states_flag == True:
            see documentation for solveSingle(args) in Solvers.py
 
    '''

    def solveDBD(v):
        
        '''
        Purpose: Solve ODEs for the DBD dose response 
        
        Inputs: 
           v: a list of lists defining the doses and parameters for this run
               v = [[DBD dose, AD dose, L dose], [e, b, k_bind, m, km, n]]
        
        Outputs: 
            doses_dbd: a list of floats corresponding to the x-axis values in the training data
                plots (DBD doses)
            sol: a list of floats corresponding to the y-axis values in the training data plots 
                (normalized reporter expression across DBD doses)   
        '''
        
        tp1_ = 18
        tp2_ = 24
       
        sol = []
        for dose in doses_dbd:
            v[0][0] = dose
            output = ''
            args =  [0, v, tp1_, tp2_, output, model]
            rep = solveSingle(args)
            sol.append(rep)
        return doses_dbd, sol     
    
    #Define v
    tp1_ = 18
    tp2_ = 24
    [e, b, k_bind, m, km, n] = p
    v = [[], []]
    v[0] = [50, 50, 100] #set base case doses
    saturating_ligand_dose = 100 * e
    v[1] = p
    
    if data == 'ligand dose response only' or data == 'ligand dose response and DBD dose response':
        doses_ligand = [0] + list(np.logspace(0, 2, 10))
        sol = []
        for dose in doses_ligand:
            v[0][2] = dose * e

            if save_internal_states_flag == True and dose == doses_ligand[-1]:
                output = 'save internal states'
                args =  [0, v, tp1_, tp2_,  output, model]
                t_1, t_2, solution_before, solution_on, state_labels = solveSingle(args)

            else:
                output = ''
                args =  [0, v, tp1_, tp2_, output, model]
                rep = solveSingle(args)   
                sol.append(rep)
      
        solutions = [i/np.max(sol) for i in sol]  
        doses_ = doses_ligand
            
    
    if data == 'ligand dose response and DBD dose response':
        norm_sol_ligand = solutions
        solutions_dbd_raw = []
        v[0][2] = saturating_ligand_dose
        
        #Solve for results of DBD plasmid dose response
        for dose_AD in [20, 10]:
            v[0][1] = dose_AD
            doses_DBD_, sol = solveDBD(v)
            solutions_dbd_raw.append(sol)
    
        #Normalize to max value in this dataset
        max1 = max(solutions_dbd_raw[0])
        max2 = max(solutions_dbd_raw[1])
        max_val = max(max1,max2)
        norm_sol_dbd_20 = [i/max_val for i in solutions_dbd_raw[0]]
        norm_sol_dbd_10 = [i/max_val for i in solutions_dbd_raw[1]]
        solutions = norm_sol_ligand + norm_sol_dbd_20 + norm_sol_dbd_10
        
        #Check for Nan and set chi2 to arbitrary, very high value if Nan in solutions
        for item in solutions:
            if math.isnan(item):
                print('Nan in solutions')
                Rsq = 0
                chi2 = 1000000
                return doses_, solutions, chi2, Rsq
  
        doses_ = [doses_ligand, doses_dbd, doses_dbd] 
    
    #Calculate the cost function
    chi2 = calcChi2(exp_data, solutions, error)
    
    if save_internal_states_flag:
        return t_1, t_2, solution_before, solution_on, state_labels
    else:
        return doses_, solutions, chi2

def solvePar(row):
    '''
    Purpose: Solve ODEs for the parameters defined in row (called directly 
                                                           by multiprocessing function)
    
    Inputs: 
        row: a row of a df containing the parameters for this set of simulations
        
    Outputs: 
        if data_type == 'PEM evaluation':
            output: a list of floats defining the chi2 value for each PEM evaluation data set 
                (length = # PEM evaluation data_sets)
            
        else:
            output:a singel item list of floats defining the chi2 value for the training data 
                (length = 1)
   '''

    #Define parameters and solve ODEs
    p = [row[1], row[2], row[3], row[4], row[5], row[6]]
    doses, norm_solutions, chi2 = solveAll(p, exp_data)
    output = [chi2]
   
    #If this is a PEM evluation run, need to calculate the cost function for each PEM evaluation data set 
    if data_type == 'PEM evaluation':
        output = []
        for k in range(0, numDatasets):
            exp = defineExpPemEval(df_pem_eval[k], data)
            sim = norm_solutions
            chi2 = calcChi2(exp, sim, error)
            output.append(chi2)
            
    return output

def optPar(row): 
    '''
    Purpose: Perform optimization with initial guesses as defined in row (called directly by 
       multiprocessing function)
    
    Inputs: 
        row: a row of a df containing the parameters and other conditions for this optimization run
        
    Outputs: 
        results_row: a list of floats, strings, and lists containing the optimization results
        results_row_labels: a list of strings containing the labels to go along with results_row
   '''
    #Unpack row
    count = row[0] + 1
    p = row[-4]
    exp_data = row[-2]
    fit_param_labels = row[-3]
    bounds = row[-1]

    #Drop index 0 and 1 (count)
    row = row[2:]
    
    #Initialize list to keep track of CF at each function evaluation
    chi2_list = []

    def solveForOpt(x, p1, p2, p3, p4, p5, p6):
        #This is the function that is solved at each step in the optimization algorithm
        
        p = [p1, p2, p3, p4, p5, p6]
        doses, norm_solutions, chi2 = solveAll(p, exp_data)
        chi2_list.append(chi2)
        
        return np.array(norm_solutions)
        
    #Set default values for min and max bounds and whether the parameter is free or not
    bound_min_list = [0] * (len(all_param_labels))
    bound_max_list = [np.inf] * (len(all_param_labels))
    
    #True if parameter is free, False if parameter is fixed
    vary_list = [False] * (len(all_param_labels)) 

    #Set min and max bounds and vary_index by comparing free parameters lists with list of all parameters
    for param_index in range(0, len(all_param_labels)): #for each param in p_all
        for fit_param_index in range(0, len(fit_param_labels)): #for each fit param
        
            #if param is fit param, change vary to True and update bounds
            if all_param_labels[param_index] == fit_param_labels[fit_param_index]: 
                vary_list[param_index] = True
                bound_min_list[param_index] = 10 ** bounds[fit_param_index][0]
                bound_max_list[param_index] = 10 ** bounds[fit_param_index][1]          
   
    #Add parameters to the parameters class
    params = Parameters()
    for index_param in range(0, len(all_param_labels)):
        params.add(all_param_labels[index_param], value=p[index_param], 
                   vary = vary_list[index_param], min = bound_min_list[index_param], 
                   max = bound_max_list[index_param])

    #Set conditions
    method = 'leastsq'
    
    #Define model
    model_ = Model(solveForOpt, nan_policy='propagate')
        
    #Perform fit and print results
    weights_ = [1/i for i in error] 
    results = model_.fit(exp_data, params, method=method, x=x, weights = weights_)
    
    #If not in the PL section, print to the console after each optimization round is complete 
    #(printing this in the PL clogs up the console because so many optimization rounds must be
    #completed)
    if PL_ID == 'no':
        print('Optimization round ' + str(count) + ' complete.')
   
    #add initial parameters to row for saving 
    result_row = p
    result_row_labels = real_param_labels_all
    
    #Define best fit parameters (results of optimization run)   
    best_fit_params = results.params.valuesdict()
    best_fit_params_list = list(best_fit_params.values())
    
    #Solve ODEs with final optimized parameters, calculate chi2, and add to result_row for saving
    doses_ligand, norm_solutions, chi2  = solveAll(best_fit_params_list, exp_data)
    result_row.append(chi2)
    result_row_labels.append('chi2')
    
    #append best fit parameters to results_row for saving (with an * at the end of each parameter name
    #to denote that these are the optimized values)
    for index in range(0, len(best_fit_params_list)):
        result_row.append(best_fit_params_list[index])
        fit_param_label = real_param_labels_all[index] + '*'
        result_row_labels.append(fit_param_label)
    
    #Define other conditions and result metrics and add to result_row for saving
    #Note that results.redchi is the chi2 value directly from LMFit and should match the chi2
    #calculated in this code if the same cost function as LMFit is used
    items = [method, results.redchi, results.success, model, 
             chi2_list, norm_solutions]
    item_labels = ['method', 'redchi2',  'success', 'model', 'chi2_list', 
                   'Simulation results']
    for i in range(0, len(items)):
        result_row.append(items[i])
        result_row_labels.append(item_labels[i])
        
    result_row = result_row[:19]
    result_row_labels = result_row_labels[:19]

    return result_row, result_row_labels


def plotTrainingDataFits(df):
    '''
    Purpose:
    1) plot simulations for all parameter sets on the same plot, with training data
    2) plot simulations for only the best case parameter set (lowest chi2), with training data
       
    Inputs: 
        df: a df containing the results of the parameter estimation method
        
    Outputs: None
    
    Figures: 
        'FITS.svg' (plot of all fits to training data with optimized parameters, 
                    # fits = #initial guesses)
        'BEST FIT.svg' (plot of best fit to training data (lowest chi2))
   '''
    #Set marker type, size of plots, and colors based on type of data
    if data_type == 'PEM evaluation': #module 1
        marker_ = '^'
        size_ = (2,2)
        color_ = pink_
        
    elif data_type == 'experimental': #module 2
        marker_ = 'o'
        size_ = (3,3)
        color_ = 'black'
        
    if data == 'ligand dose response only': 
        dose_responses = list(df['Simulation results'])
        fig = plt.figure(figsize = size_)
        ax1 = plt.subplot(111)  
        
        #Plot experimental data
        ax1.errorbar(doses_ligand, exp_data, color = module_color, marker = marker_, yerr = error, 
                     fillstyle = 'none', linestyle = 'none',capsize = 2, label = 'Training data')

        #Plot simulated data for each parameter set in df
        sns.set_palette("Greys", len(dose_responses))
        count = 0
        for data_ in dose_responses:
            count += 1
            ax1.plot(doses_ligand, data_, linestyle = ':', marker = None, label = 'Model fit ' 
                     + str(count))
        ax1.set_xscale('symlog')
        ax1.set_xlabel('Ligand dose (nM)')
        ax1.set_ylabel('Reporter expression')
        plt.savefig('./FITS.svg', bbox_inches="tight")
        
        #Plot best case parameter set ONLY 
        fig = plt.figure(figsize = size_)
        ax1 = plt.subplot(111)  
        
        chi2_list = list(df['chi2'])
        val, idx = min((val, idx) for (idx, val) in enumerate(chi2_list))
        best_case = dose_responses[idx]
        
        #Plot experimental data
        ax1.errorbar(doses_ligand, exp_data, color = module_color, marker = marker_, yerr = error,  
                     fillstyle = 'none', linestyle = 'none',capsize = 2, label = 'Training data')

        #Plot simulated data for the best case parameter set
        ax1.plot(doses_ligand, best_case,  color = color_, linestyle = ':', marker = None, 
                 label = 'Best case model fit')
        ax1.set_xscale('symlog')
        ax1.set_xlabel('Ligand dose (nM)')
        ax1.set_ylabel('Reporter expression')
        plt.savefig('./BEST FIT.svg', bbox_inches="tight")
    
    
    elif data == 'ligand dose response and DBD dose response': 
        fig = plt.figure(figsize = (9,3))
        fig.subplots_adjust(wspace=0.2)
        ax1 = plt.subplot(131)   
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)
        
        dose_responses = list(df['Simulation results'])
        chi2_list = list(df['chi2'])
        val, idx = min((val, idx) for (idx, val) in enumerate(chi2_list))
        best_case = dose_responses[idx]
        
        L_exp = exp_data[:11]
        DBD20_exp = exp_data[11:19]
        DBD10_exp = exp_data[19:]
        
        L_sim = best_case[:11]
        DBD20_sim = best_case[11:19]
        DBD10_sim = best_case[19:]
        
        e1 = error[:11]
        e2 = error[11:19]
        e3 = error[19:]
    
        colors = ['black', 'black', 'black']
        linestyles = ['dotted', 'dotted', 'dotted']
        ax1.errorbar(doses_ligand, L_exp, color = colors[0], marker = marker_, yerr = e1, 
                     fillstyle = 'none', linestyle = 'none',capsize = 2)
        ax2.errorbar(doses_dbd, DBD20_exp, color =  colors[1], marker = marker_, yerr = e2,  
                     fillstyle = 'none', linestyle = 'none',capsize = 2)
        ax3.errorbar(doses_dbd, DBD10_exp, color =  colors[2], marker = marker_, yerr = e3,  
                     fillstyle = 'none', linestyle = 'none',capsize = 2)
        ax1.set_xscale('symlog')
     
        count = 0
        sns.set_palette("Greys", len(dose_responses))
        for dose_response in dose_responses:
            count += 1
            ax1.plot(doses_ligand, dose_response[0:11], marker = None, label = 'Model fit ' + str(count), 
                     linestyle = linestyles[0], color = colors[0])
            ax2.plot(doses_dbd, dose_response[11:19], marker = None, label = 'Model fit ' + str(count), 
                     linestyle = linestyles[1], color = colors[1])
            ax3.plot(doses_dbd, dose_response[19:], marker = None, label = 'Model fit ' + str(count), 
                     linestyle = linestyles[2], color = colors[2])
    
        ax1.set_xscale('symlog')
        ax1.set_xlabel('Ligand dose (nM)')
        ax1.set_ylabel('Reporter expression')
        ax2.set_xlabel('DBD plasmid (ng)')
        ax2.set_ylabel('Reporter expression')
        ax3.set_xlabel('DBD plasmid (ng)')
        ax3.set_ylabel('Reporter expression')
        plt.savefig('./FITS.svg', bbox_inches="tight")
     
        fig = plt.figure(figsize = (6,3))
        fig.subplots_adjust(wspace=0.2)
        ax1 = plt.subplot(121)   
        ax2 = plt.subplot(122)
        ax1.set_xscale('symlog')
        
        colors = ['black', 'black', 'dimgrey']
        ax1.errorbar(doses_ligand, L_exp, color = colors[0], marker = marker_, yerr = e1, 
                     fillstyle = 'none', linestyle = 'none',capsize = 2)
        ax2.errorbar(doses_dbd, DBD20_exp, color =  colors[1], marker = marker_, yerr = e2, 
                     fillstyle = 'none', linestyle = 'none',capsize = 2)
        ax2.errorbar(doses_dbd, DBD10_exp, color =  colors[2], marker = marker_, yerr = e3, 
                     fillstyle = 'none', linestyle = 'none',capsize = 2)
  
        #Plot simulated data for the best case parameter set
        ax1.plot(doses_ligand, L_sim, linestyle =':', marker = None, 
                 label = 'Model fit ' + str(count), color = colors[0])
        ax2.plot(doses_dbd, DBD20_sim, linestyle = ':', marker = None, 
                 label = 'Model fit ' + str(count), color = colors[1])
        ax2.plot(doses_dbd, DBD10_sim, linestyle = ':', marker = None, 
                 label = 'Model fit ' + str(count), color = colors[2])
    
        #Set x and y labels
        ax1.set_xlabel('Ligand dose (nM)')
        ax1.set_ylabel('Reporter expression')
        ax2.set_xlabel('DBD plasmid (ng)')
        ax2.set_ylabel('Reporter expression')
        plt.savefig('./BEST FIT.svg', bbox_inches="tight")
        
# =============================================================================
#  Module 2 - code for parameter estimation using training data
# =============================================================================  
def plotParamDistributions(df):
    '''
    Purpose: Plot parameter distributions across initial guesses following parameter estimation with 
        training data (Module 2)
    
    Inputs: 
        df: dataframe containing the results of the PEM
        
    Outputs: None
    
    Figures: 
        './FITS Rsq ABOVE 0.99.svg' (plot of training data and simulated data for 
             parameter sets with Rsq > = 0.99)
        'OPTIMIZED PARAMETER DISTRIBUTIONS.svg' (plot of parameter distributions for 
             parameter sets with Rsq > = 0.99)
   '''
   
    #Only keep rows for which Rsq >= .99
    df = df[df["Rsq"] >= 0.99]
    dose_responses = list(df['Simulation results'])
    
    # =============================================================================
    # 1. dose response for parameter sets with Rsq > .99
    # ============================================================================
    fig = plt.figure(figsize = (3,3))
    ax1 = plt.subplot(111)  
    error = [.05] * 11
    
    #Plot experimental/training data
    ax1.errorbar(doses_ligand, exp_data[:11], color = 'black', marker = 'o', yerr = error,  
                 fillstyle = 'none', linestyle = 'none',capsize = 2, label = 'Training data')
    ax1.set_xscale('symlog')
    
    #Plot simulated data for each parameter set in df
    n = len(dose_responses)
    color = plt.cm.Blues(np.linspace(.1, 1, n))
    plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    count = 0
    for dose_response in dose_responses:
        count += 1
        ax1.plot(doses_ligand, dose_response, linestyle = ':', marker = None, 
                 label = 'Model fit ' + str(count))
    ax1.set_xlabel('Ligand dose (nM)')
    ax1.set_ylabel('Reporter expression')
    plt.savefig('./FITS Rsq ABOVE 0.99.svg', bbox_inches="tight")
    
    # =============================================================================
    # 2. parameter distributions for parameter sets with Rsq > .99
    # =============================================================================
    param_labels = ['e*', 'b*', 'k_bind*', 'm*', 'km*', 'n*']
    for label in param_labels:
        new_list = [log10(i) for i in list(df[label])]
        df[label] = new_list
    
    plt.subplots(1,1, figsize=(4,3), sharex = True)
    df = pd.melt(df, id_vars=['Rsq'], value_vars=param_labels)
    ax = sns.boxplot(x='variable', y='value', data=df, color = sky_blue)
    ax = sns.swarmplot(x='variable', y='value', data=df, color="gray")
    ax.set(xlabel='Parameter', ylabel='log(value)')
    plt.savefig('OPTIMIZED PARAMETER DISTRIBUTIONS.svg', dpi = 600)

def runParameterEstimation():
    '''
    Purpose: Run PEM (global search, filter, and optimization)
    
    Inputs: 
      
    Outputs:   
        df: dataframe containing the results of the PEM
        best_case_params: a list of floats containing the best case parameter 
            values found by the PEM (lowest chi2)
    
    Files: 
        'GLOBAL SEARCH RESULTS.xlsx' (dataframe containing results of the global search)
        'OPT RESULTS.xlsx' (dataframe containing results of the optimization algorithm)
       
    '''
    # =============================================================================
    # 1. Global search
    # =============================================================================
    
    #Generate parameter sets
    df_params = generateParams(problem_free, n_search, p_all, problem_all_params, model)
    print('Starting global search...')
    
    if parallelization == 'no':  # without multiprocessing
        output = []
        for row in df_params.itertuples(name = None):
            result = solvePar(row)
            output.append(result)

    elif parallelization == 'yes':  # with multiprocessing
        with mp.Pool(conditions_dictionary["num_cores"]) as pool:
            result = pool.imap(solvePar, df_params.itertuples(name = None))
            pool.close()
            pool.join()
            output = [[round(x[0],4)] for x in result]
    
    #Unpack global search results and save as df
    chi2_list = []
    for pset in range(0, len(output)):
        chi2_list.append(output[pset][0])
    df_results = df_params
    df_results['chi2'] = chi2_list
    
    filename = 'GLOBAL SEARCH RESULTS'
    with pd.ExcelWriter(filename + '.xlsx') as writer:  # doctest: +SKIP
        df_results.to_excel(writer, sheet_name='GS results')
    print('Global search complete.')
   
    # =============================================================================
    # 2. Filter
    # =============================================================================
    
    #Filter results of global search
    filtered_data, initial_guesses = filterGlobalSearch(df_results, n_initial_guesses, 
                                                        all_param_labels, 'chi2')
    print('Filtering complete.')
    
    # =============================================================================
    # 3. Optimization
    # =============================================================================
    print('Starting optimization...')
    #Define df with optimization conditions
    df = initial_guesses
    df['exp_data'] = [exp_data] * len(df.index)
    list_ = [problem_free['bounds']] * len(df.index)
    df['bounds'] = list_
    all_opt_results = []
    
    if parallelization == 'no':   # without multiprocessing
        for row in df.itertuples(name = None):
            result_row, result_row_labels = optPar(row)
            all_opt_results.append(result_row)
      
    elif parallelization == 'yes':  # with multiprocessing
        with mp.Pool(num_cores) as pool:
            result = pool.imap(optPar, df.itertuples(name=None))
            pool.close()
            pool.join()
            output = [[list(x[0]), list(x[1])] for x in result]
            
        for ig in range(0, len(output)):
            all_opt_results.append(output[ig][0])
            result_row_labels = output[ig][1]
            
    print('Optimization complete.')
    
    #Save results of the optimization rounds
    df_opt = pd.DataFrame(all_opt_results, columns = result_row_labels)
   
    #Plot training data and model fits with optimized parmaeters
    plotTrainingDataFits(df_opt)
    
    #Sort by chi2
    df = df_opt.sort_values(by=['chi2'], ascending = True)
    
    #Save best case calibrated parameters (lowest chi2)
    best_case_params = []
    for i in range(0, len(p_all)):
        col_name = real_param_labels_all[i] + '*'
        val = df[col_name].iloc[0]
        best_case_params.append(round(val, 4))
      
    #Calculate Rsq for optimized parameter sets
    Rsq_list = []
    for j in range(0, n_initial_guesses):
        params = []
        for i in range(0, len(p_all)):
            col_name = real_param_labels_all[i] + '*'
            val = df[col_name].iloc[j]
            params.append(val)
        doses, norm_solutions, chi2 = solveAll(params, exp_data)
        Rsq = calcRsq(norm_solutions, exp_data)  
        Rsq_list.append(Rsq)
    df['Rsq'] = Rsq_list
   
    #Plot parameter distributions across optimized parameter sets
    if data == 'ligand dose response only':
        plotParamDistributions(df)

    #Save results of the optimization rounds
    filename = 'OPT RESULTS'
    with pd.ExcelWriter(filename + '.xlsx') as writer:  # doctest: +SKIP
        df.to_excel(writer, sheet_name='OPT Results')
    
    '''FIT CRITERIA'''
    print('*************************')
    print('')
    print('Calibrated parameters: ' + str(best_case_params))
    print('')
    Rsq_opt_max = round(df['Rsq'].iloc[0], 3)
    print('Rsq = ' + str(Rsq_opt_max))
    print('')
    chi2opt_min = round(df['chi2'].iloc[0], 3)
    print('chi2 = ' + str(chi2opt_min))
    print('')
    print('*************************')

    return df, best_case_params

# =============================================================================
#  Module 3 - code to run parameter identifiability analysis with the parameter
#  profile likelihood
# =============================================================================

def calcPL(calibrated_vals, calibrated_chi2, threshold_val):
    
    '''
    Purpose: Evaluate the profile likelihood for each free parameter.
    
    Input: 
        calibrated_vals: a list of calibrated parameter values (starting point for PPL calculations)
        calibrated_chi2: a float defining the value of chi2 for the calibrated 
            parameter set (calibrated_vals)
        threshold_val: a float defining the threshold value of chi2
    
    Outputs: 
        df_list: a list of dataframes, each holding the PPL results for a single parameter 
            (length = # free parameters)
    
    '''

    print('Starting parameter identifiability analysis...')
    
    def calcPL_Single(param_vals, fixed_index, fixed_val, free_index):
        '''
        Purpose: Calculate chi2_PL for a single datapoint on the profile likelihood
        
        Inputs: 
            param_vals: a list of parameter values (initial guesses)
            fixed_index: an integer defining the index in param_vals that should be fixed for 
                this calculation, this is the index of the parameter in the list of ALL 
                parameters (real_param_labels_all)
            fixed_val: a float defining the value that the fixed index should be fixed at
            free_index: an integer defining the index in the list of FREE parameters 
                (real_param_labels_free) that should be fixed for this calculation
            
        Outputs: 
           fixed_val: a float defining the value that the fixed index should be fixed at 
               (x-axis value of the datapoint for PPL plots)
           minchi2: a float defining the minimum chi2 attainable for the parameter 
               estimation problem (x-axis value of the datapoint for PPL plots)
           best_case_params: a list of flats defining the parameter set for which the minchi2 
               was obtained
        
        Files:
            './PROFILE LIKELIHOOD RESULTS ' + param + '.xlsx' (PPL results for each free parameter, 
                                                               # files = # free parameters)
        
        '''
        
        '''Part 1: Define the parameter estimation problem'''
        #Initialize parameter estimation problem 
        problem_ = {'num_vars': 0,  #set free parameters and bounds
                    'names': [], 
                    'bounds': []} #bounds are in log scale
    
        #Define parameter estimation problem (fix parameter at fixed index, 
        #let other parameters vary)
        problem_['num_vars'] = num_vars - 1
        problem_['names'] = [x for i,x in enumerate(problem_free['names']) if i != free_index]
        problem_['bounds'] = [x for i,x in enumerate(problem_free['bounds']) if i != free_index]
        
        '''Part 2: global search and filter to define initial guesses for optimization'''
        #Generate parameter sets to sample
        df_params = generateParams(problem_, n_search, param_vals, problem_all_params, model)

        #Global search
        if parallelization == 'no': #without multiprocessing
            output = []
            for row in df_params.itertuples(name = None):
                result = solvePar(row)
                output.append(result)
        elif parallelization == 'yes':  #with multiprocessing
            with mp.Pool(8) as pool:
                result = pool.imap(solvePar, df_params.itertuples(name = None))
                pool.close()
                pool.join()
                output = [[round(x[-1],4)] for x in result]    
                
        df_results = df_params
        df_results['chi2'] = output
        filtered_data, initial_guesses  = filterGlobalSearch(df_results, n_initial_guesses, 
                                                             all_param_labels, 'chi2')
        '''Part 3: optimization'''
        df = initial_guesses
        all_opt_results = []
        df['exp_data'] = [exp_data] * len(df.index)
        list_ = [problem_['bounds']] * len(df.index)
        df['bounds'] = list_
    
        if parallelization == 'no': #without multiprocessing
            for row in df.itertuples(name = None):
                result_row, result_row_labels = optPar(row)
                all_opt_results.append(result_row)
                
        elif parallelization == 'yes': #with multiprocessing
            with mp.Pool(8) as pool: #with multiprocessing
                result = pool.imap(optPar, df.itertuples(name=None))
                pool.close()
                pool.join()
                output = [[list(x[0]), list(x[1])] for x in result]
            for ig in range(0, len(output)):
                all_opt_results.append(output[ig][0])
                result_row_labels = output[ig][1] 
                
        df_opt = pd.DataFrame(all_opt_results, columns = result_row_labels)
        minchi2 = min(list(df_opt['chi2']))
        df = df_opt.sort_values(by=['chi2'], ascending = True)

        #Save best calibrated parameter set (lowest chi2)
        best_case_params = []
        for i in range(0, len(real_param_labels_all)):
            col_name = real_param_labels_all[i] + '*'
            val = df[col_name].iloc[0]
            best_case_params.append(val)
          
        #Save results
        filename = run + ' OPT RESULTS PL'
        with pd.ExcelWriter(filename + '.xlsx') as writer:  # doctest: +SKIP
            df.to_excel(writer, sheet_name='OPT Results')
       
        return fixed_val, minchi2, best_case_params

    
    #Determine index (or indicies) for PPL based on param_index_PL
    #These are the indicies of the free parameters in the list real_param_labels_all
    indicies = []  
    for i, value in enumerate(p_all):
        label = real_param_labels_all[i]
        if label in real_param_labels_free:
            indicies.append(i)
 
    #Initialize lists to hold parameter and chi2_PL values
    #value of fixed parameter
    param_lists_fixed = [] 
    
    #value of chi2_PL
    chi2_PL_lists = [] 
    
    #time to complete PL analysis, per parameter
    time_list = [] 
    
    #lists of all parameter values for each accepted chi2_PL calculation
    param_lists_all = [] 
    
    #list of dataframes, each holding results for different parameters
    df_list = [] 
    
    #number of evaluations (full PEM runs)
    num_evals = 0
    
    #Set the minimum and maximum acceptable step choices (x2_PL(current step) - x2_PL(last step))
    #set the min step to 1% of starting value
    min_step_list = [.01 * i for i in calibrated_vals] 
    
    #set the max step to 20% of starting value
    max_step_list = [.2 * i for i in calibrated_vals] 
    
    #Set the target value for PL difference between steps (chi2_PL(current step) - chi2_PL(last step)) 
    #and acceptable range
    q = .1 
    target_val = threshold_val * q
    min_threshold_lim = target_val * .1
    max_threshold_lim = target_val * 2
   
    #for each parameter, evaluate the PPL
    for fixed_index in indicies:   
        #Set variables associated with calculating PL
        #maximum number of steps for EACH direction
        max_steps = 50
        
        #Record the start time
        startTime = datetime.datetime.now()
        
        #Determine the index in real_param_labels_free that corresponds to the index in real_param_labels_all
        fixed_label = real_param_labels_all[fixed_index]
        for j, val in enumerate(real_param_labels_free):
            if val == fixed_label:
                free_index = j #The index of the fixed parameter (for this PL run) in the list of free parameters
   
        #Set min and max flags to default values 
        min_flag = False
        max_flag = False
        
        #Define the min and max bounds of the fixed parameter, convert to linear scale
        min_bound_log = problem_free['bounds'][free_index][0]
        max_bound_log = problem_free['bounds'][free_index][1]
        min_bound = 10 ** min_bound_log
        max_bound = 10 ** max_bound_log
        
        print('******************')
        print('Starting profile likelihood calculations for ' + 
              real_param_labels_all[fixed_index] + '...')
        
        #for each direction, evaluate PPL (-1 is LHS and 1 is RHS)
        for direction in [-1, 1]: 
            print('')
            print('direction: ' + str(direction))
            
            #Set min and max step vals for the fixed parameter
            min_step = min_step_list[fixed_index] 
            max_step = max_step_list[fixed_index] 
            
            #if this is k_bind or m in the -1 direction, decrease min_step by 2 orders of magnitude
            if fixed_index == 2 and direction == -1:
                min_step = .01 * min_step #kbind
            if fixed_index == 3 and direction == -1:
                min_step = .01 * min_step #m
            if fixed_index == 3 and direction == 1:
                max_step = 3 * max_step #m
                max_steps = 100
    
            print('min step: ' + str(min_step))
            print('max step: ' + str(max_step))
    
            #Initialize lists to hold results for this PPL only
            chi2_PL_list = [] #chi2_PL values
            param_list = []  #fixed parameter values, linear scale
            params_all_list = [] #lists of all parameter values
           
            #Initialize the step number counter and attempt_number counters
            step_number = 0
            attempt_number = 0

            while step_number < max_steps:
                print('******************')
                print('step_number: ' + str(step_number))
                print('attempt_number: ' + str(attempt_number))
                print('')
        
                #If this is the first step
                if step_number == 0: 
                    
                    #Set the fixed value as the calibrated value
                    fixed_val = calibrated_vals[fixed_index]
                    print('starting val: ' + str(fixed_val))
                    
                    #Set the min_ for the binary step to 0 and the max_ to the max_step
                    min_ = 0
                    max_ = max_step
                    
                    print('min_: ' + str(min_))
                    print('max_: ' + str(max_))
                    
                    #Set the min and max flags to false
                    min_flag = False
                    max_flag = False
               
                #If this is not the first step
                else: 
                    
                    #If the step value is more than or equal to the max_step, 
                    #then replace the step with #the max step and set max_flag to True
                    if step_val >= max_step: 
                        step_val = max_step
                        max_flag = True
                        print('step replaced with max step')
                       
                    #If the step value is less than or equal to the min_step, 
                    #then replace the step with the min step and set min_flag to True
                    elif step_val <= min_step: 
                        step_val = min_step
                        min_flag = True
                        print('step replaced with min step')
                      
                    #If the step is between the min and max step limits, set flags to 
                    #False and continue
                    else:
                        min_flag = False
                        max_flag = False
                     
                    #Determine the next fixed value
                    fixed_val = param_list[step_number - 1] + (direction * step_val)
                    print('previous val: ' + str(round(param_list[step_number - 1], 8))) 
                    print('step val: ' + str(round(step_val, 8)))
                    print('fixed val: ' + str(round(fixed_val)))
                    
                    if fixed_val <= min_bound: #if fixed val is negative or less than the min bound of the parameter
                        print('fixed val is negative')
                        
                        #if this is the minimum step, break (cannot take a smaller step 
                        #to reach a > 0 fixed value)    
                        if min_flag == True: 
                            print('negative or zero fixed value reached - break')
                            break #out of top level while loop and start next direction/parameter
                        
                        #if this is not the minimum step, try with half the last step
                        else: 
                            
                            
                            while fixed_val <= min_bound:
                                print('try again')
                                #reset the max step to be half what it was before
                                max_step = max_step/2 
                                
                                #try a step val half the size as the previous step
                                step_val = step_val/2 
                                
                                #determine fixed value
                                fixed_val = param_list[step_number - 1] + (direction * step_val) 
                                print('fixed val: ' + str(fixed_val))
                                
                                if step_val <= min_step:
                                    step_val = min_step
                                    fixed_val = param_list[step_number - 1] + (direction * step_val) 
                                    min_flag = True
                                    print('step replaced with min step')
                                    break #break out of this while loop
                                
                    if fixed_val <= 0.0:
                        print('negative or zero fixed value reached - break')
                        break #out of top level while loop and start next direction/parameter
                          
   
                    print('new val: ' + str(round(fixed_val, 4))) #linear
                
                #Calculate PL
            
                #Restructure parameter set to feed into calcPL_Single
                params_for_opt = p_all
                for i in range(0, len(real_param_labels_all)): #for each parameter in p_all
                    for j in range(0, len(real_param_labels_free)): #for each free parameter
                    
                        #if parameter is a free parameter, replace with calibrated 
                        if real_param_labels_all[i] == real_param_labels_free[j]: 
                            if i == fixed_index:
                                params_for_opt[i] = fixed_val #replace with fixed val
                            else:   
                                params_for_opt[i] = calibrated_vals[i] #Replace with cal val
               
                param_val, chi2_PL_val, param_vals = calcPL_Single(params_for_opt, 
                                                                   fixed_index, fixed_val, free_index)
                print('chi2_PL: ' + str(round(chi2_PL_val, 3)))
                num_evals += 1
                print('calibrated parameters: ' + str(param_vals))
                
                #Determine whether to accept step or try again with new step size
    
                #if this is not the first step (calibrated parameter value)
                if step_number != 0:
                    
                    #Calculate difference in PL between current value and previous value
                    PL_diff = abs(chi2_PL_val - chi2_PL_list[step_number -1])
                    print('PL_difference: ' + str(round(PL_diff, 3)))
                    print('')
                   
                    #if PL_diff is in between the min and max limits for PL difference
                    if PL_diff <= max_threshold_lim and PL_diff >= min_threshold_lim: 
                        
                        #if the PL value is greater than or equal to the threshold value, but less 
                        #than 1.1 * the threshold value
                        if chi2_PL_val >= threshold_val and chi2_PL_val <= 1.1 * threshold_val: 
                            
                            #Record values and break loop 
                            chi2_PL_list.append(chi2_PL_val)
                            param_list.append(param_val)
                            params_all_list.append(param_vals)
                            print('break - Threshold chi2 reached')
                            break
                        
                        #otherwise, if the PL value is greater than 1.1 * the threshold value, 
                        #then the step is too large
                        elif chi2_PL_val > 1.1 * threshold_val:
                            
                            #If this is the minimum step, then a smaller step cannot be taken
                            if min_flag == True:
                                
                                #Record values and break loop 
                                chi2_PL_list.append(chi2_PL_val)
                                param_list.append(param_val)
                                params_all_list.append(param_vals)
                                print('break - Threshold chi2 reached')
                                break
                              
                            #If this is not the minimum step, then a smaller step should be used
                            else:
                                #Set the max bound for the binary step to be equal to 
                                #the current step
                                max_ = step_val
                                
                                #Calculate the next step value (binary step algorithm)
                                step_val  = (min_ + max_)/2
                                
                                #increase the attempt number
                                attempt_number += 1
                                
                                print('step rejected - too large')  
                                print('')
                                print('new bounds')
                                print('min_: ' + str(min_))
                                print('max_: ' + str(max_))
                                print('new step val: ' + str(step_val))
                        
                        #Otherwise, if the fixed parameter hits the min or max bound
                        elif param_val > max_bound or param_val < min_bound: 
                            
                            #Record values and break loop 
                            chi2_PL_list.append(chi2_PL_val)
                            param_list.append(param_val)
                            params_all_list.append(param_vals)
                            print('break - Parameter bound reached')
                            break
                        
                        #Otherwise, accept the step and record the results 
                        #(the parameter is not above the threshold and does not 
                        #reach the parameter bound)
                        else: 
                            print('step accepted')
                            
                            #Record results
                            chi2_PL_list.append(chi2_PL_val)
                            param_list.append(param_val)
                            params_all_list.append(param_vals)
                
                            #increase the step number
                            step_number += 1
                            
                            #Reset the attempt counter
                            attempt_number = 0
                            
                            #Set min_ bound back to 0 (no step from the previously recorded value)
                            min_ = 0
                    
                    #if the step size is the maximum step and the PL difference is still too low
                    elif max_flag == True and PL_diff < min_threshold_lim: 
                       
                        #If the parameter value is above the max bound or below the min bound
                        if fixed_val > max_bound or fixed_val < min_bound:
                        
                            #Record results and break loop
                            chi2_PL_list.append(chi2_PL_val)
                            param_list.append(param_val)
                            params_all_list.append(param_vals)
                            print('break - Parameter bound reached')
                            break
                        
                        #if the PL value is more than or equal to the threshold value and is less 
                        #than 1.1 * the threshold value
                        elif chi2_PL_val >= threshold_val and chi2_PL_val <= 1.1 * threshold_val: 
                            
                            #Record results and break loop
                            chi2_PL_list.append(chi2_PL_val)
                            param_list.append(param_val)
                            params_all_list.append(param_vals)
                            print('break - Threshold chi2 reached')
                            break
                        
                        #otherwise, if the step size is the maximum step AND the PL difference 
                        #is still too low AND the threshold value is not met AND and the 
                        #parameter bounds are not reached, then accept the step by default
                        else: 
                            #Record results
                            chi2_PL_list.append(chi2_PL_val)
                            param_list.append(param_val)
                            params_all_list.append(param_vals)
                            
                            #Increasethe step counter
                            step_number += 1
                            
                            #Reset the attempt counter
                            attempt_number = 0
            
                            print('step accepted by default - max step value reached')
                            
                            #Set min_ bound for step size calculations equal 0 (no step)
                            min_ = 0
                            
                            #Keep the step value at the maximum step value
                            step_val = max_step
                        
                    #if the parameter reaches the min step, but the PL difference is still too high
                    elif min_flag == True and PL_diff > max_threshold_lim: 
                        
                        #if the PL value is more than or equal to the threshold value and is 
                        #less than 1.1 * the threshold value
                        if chi2_PL_val >= threshold_val:  
                            
                            #Record results and break loop
                            chi2_PL_list.append(chi2_PL_val)
                            param_list.append(param_val)
                            params_all_list.append(param_vals)
                            print('break - Threshold chi2 reached')
                            print('min_step used - cannot take a smaller step')
                            break
            
                        #if parameter hits bound
                        elif fixed_val > max_bound or fixed_val < min_bound: 
                            
                            #Record results and break loop
                            chi2_PL_list.append(chi2_PL_val)
                            param_list.append(param_val)
                            params_all_list.append(param_vals)
                            print('break - Parameter bound reached')
                            print('min_step used - cannot take a smaller step')
                            break
                 
                        #if parameter is not above the threshold or the parameter bound, 
                        #accept the step and record the results by default
                        else: 
                            #Record results
                            chi2_PL_list.append(chi2_PL_val)
                            param_list.append(param_val)
                            params_all_list.append(param_vals)
                            
                            #Add 1 to the step counter
                            step_number += 1
                            
                            #Reset the attempt counter
                            attempt_number = 0
                            print('step accepted by default - min step value reached')
                            
                            #Set min_ bound for step size calculations equal to the current step val
                            min_ = 0
                            step_val = min_step
                   
                    #if conditions are not met because PL difference is too large, 
                    #try again with a smaller step size (or if a differenence is negative, 
                    #then step is too large and need to try a smaller one if possible)
                    elif PL_diff > max_threshold_lim or PL_diff < 0: 
        
                        #Set the max bound for the next step to be equal to the current step
                        max_ = step_val
                        
                        #Calculate the next step value
                        step_val  = (min_ + max_)/2
                        
                        #If min and max for binary step calculations are equal, then set the step 
                        #val to default to the min step
                        if min_ == max_: 
                            step_val = min_step
                            print('step replaced with min step')
                        
                        #add 1 to the attempt number
                        attempt_number += 1
                        
                        print('step rejected - too large')  
                        print('')
                        print('new bounds')
                        print('min_: ' + str(min_))
                        print('max_: ' + str(max_))
                        print('new step val: ' + str(step_val))
                       
                    #if conditions are not met because PL difference is too small, try again with a 
                    #larger step size
                    elif PL_diff < min_threshold_lim:
                        #Set the min bound for the next step to be equal to the current step
                        min_ = step_val 
                        
                        #Calculate the next step value
                        step_val  = (min_ + max_)/2
                        
                        #If min and max for binary step calculations are equal, then set the 
                        #step val to default to half of the max step
                        if min_ == max_:
                            step_val = max_step / 2
                            print('step replaced with max step')
                        
                        #Increase the attempt number
                        attempt_number += 1
                        
                        print('step rejected - too small')  
                        print('')
                        print('new bounds')
                        print('min_: ' + str(min_))
                        print('max_: ' + str(max_))
                        print('new step val: ' + str(step_val))

                    else:
                        print('else')
                          
                #if is the calibrated param (0th step), record results by default and start with the 
                #max step value
                elif step_number == 0: 
           
                    #Set next step to max step value
                    step_val  = max_step
                    
                    #Record results
                    if direction == -1:
                        chi2_PL_list.append(chi2_PL_val)
                        param_list.append(param_val)  
                        params_all_list.append(param_vals)
                        
                        PL_val_calibrated = chi2_PL_val
                        param_vals_calibrated = param_vals
                        
                        step_number += 1
                        attempt_number = 0
                        
                    elif direction == 1:
                        chi2_PL_list.append(PL_val_calibrated)
                        print('step accepted - calibrated parameter (Step 0)')
                        print(PL_val_calibrated)
                        param_list.append(param_val)     
                        params_all_list.append(param_vals_calibrated)
                        
                        step_number += 1
                        attempt_number = 0
                     
            #Prepare lists for plotting (reverse the negative direction (left), then append the 
            #positive direction (right))
            if direction == -1: 
                chi2_PL_list.reverse()
                PL_left = chi2_PL_list
                param_list.reverse()
                params_left = param_list
                params_all_list.reverse()
                param_vals_left = params_all_list
                
            elif direction == 1:
                PL_right = chi2_PL_list
                params_right = param_list
                param_vals_right = params_all_list
                
        
        #combine LHS and RHS of the PPL
        all_PL_vals = PL_left + PL_right 
        param_list_all = params_left + params_right
        param_vals_ = param_vals_left + param_vals_right
        
        print('fixed parameter values:')
        print(param_list_all)
        print('profile likelood values:')
        print(all_PL_vals)
        
        print('***')
        print('number of PEM rounds to evaluate PPL for this parameter: ' + str(num_evals))

        #Record stop time
        stop_time = datetime.datetime.now()
        elapsed_time = stop_time - startTime
        elapsed_time_total = round(elapsed_time.total_seconds(), 1)
        print('Time for this parameter (minutes): ' + str(round(elapsed_time_total/60, 4)))
        print('***')
        time_list.append(elapsed_time_total/60)
          
        #Initialize df to store results
        df_PL = pd.DataFrame()
        
        #Fill df with data
        param_label = real_param_labels_free[free_index]
        df_PL['fixed ' + param_label] = param_list_all
        df_PL['fixed ' + param_label + ' PL'] = all_PL_vals
        df_PL['fixed ' + param_label + ' all params'] = param_vals_
        
        #add results to full list across parameters
        param_lists_fixed.append(param_list_all)
        chi2_PL_lists.append(all_PL_vals)
        param_lists_all.append(param_vals_)
 
        #Save df
        filename = './PROFILE LIKELIHOOD RESULTS ' + real_param_labels_free[free_index] + '.xlsx'
        with pd.ExcelWriter(filename) as writer:  # doctest: +SKIP
            df_PL.to_excel(writer, sheet_name = real_param_labels_free[free_index])
    
        df_list.append(df_PL)

    #Plot the results
    plotPL(param_lists_fixed, chi2_PL_lists, threshold_val, calibrated_vals, PL_val_calibrated)

    #Record the total time 
    print('Total time (minutes): ' + str(round(sum(time_list), 4)))
    print('Average time per parameter (minutes): ' + str(round(np.mean(time_list), 4)))
    print('SD (minutes): ' + str(round(np.std(time_list), 4)))
    
    return df_list
    

def plotPL(param_lists, chi2_PL_lists, threshold_val, calibrated_vals, calibrated_chi2):
    
    '''
    Purpose: Plot PPL results (from calculations in Calc_PL)
    
    Inputs: 
        param_lists: a list of lists containing the fixed parameter values for each PPL 
            (length of inner lists = # PL evaluations for the given free parameter, 
             length of outer list = # free parameters)
        chi2_PL_lists: a list of lists containing the chi2_PL values for each datapoint 
            along the PPL (length of inner lists = # PL evaluations for the given free parameter, 
            length of outer list = # free parameters)
        threshold_val: a float defining the threshold value of chi2
        calibrated_vals: a list of calibrated parameter values (starting point for PPL calculations)
        calibrated_chi2: a float defining the value of chi2 for the calibrated parameter set 
            (calibrated_vals)
  
    Outputs: None
    
    Figures:
        'PROFILE LIKELIHOOD PLOTS.svg' (plot of profile likelihoods for each parameter)
        
    '''
    
    #Set up figure structure
    num_vars = len(real_param_labels_free)
    fig, axs = plt.subplots(nrows=1, ncols=num_vars, figsize=(num_vars*2, 2))
    fig.subplots_adjust(wspace=.3)
    fig.subplots_adjust(hspace=.3)
    axs = axs.ravel()
    for axi in axs.flat:
        axi.xaxis.set_major_locator(plt.MaxNLocator(3))
    titles = real_param_labels_free
    
    #Convert to log scale
    calibratedVals_log = [log10(i) for i in calibrated_vals]
    
    #For each parameter, plot results in a new subplot
    for i in range(0, num_vars):
        x = param_lists[i]
        y = chi2_PL_lists[i]
        ax = axs[i]
        title = titles[i]
        
        x_plot = []
        y_plot = []
        for j in range(0, len(x)):
            
            if abs(x[j]) > (10 ** -10):
                x_plot.append(x[j])
                y_plot.append(y[j])
            else:
                print(x[j])
        
        #Convert to log scale
        x_log = [log10(i) for i in x_plot]
        ax.plot(x_log, y_plot, 'o-', color = black_, markerSize = 4, fillstyle='none', zorder = 1)
        ax.set_xlabel(title)
        
        if i == 0 or i == 3:
            ax.set_ylabel('chi2')
        
        #plot calibrated parameter value as a star  
        cal_vals = []
        for i_ in range(0, len(real_param_labels_all)): #for each parameter in p_all
            for j in range(0, len(real_param_labels_free)): #for each free param
                if real_param_labels_all[i_] == real_param_labels_free[j]: #if parameter is a free parameter 
                    #replace with calibrated val
                    cal_vals.append(calibratedVals_log[i_])
        
        x = [cal_vals[i]]
        y = [calibrated_chi2]
        ax.scatter(x, y, s=16, marker='o', color = sky_blue, zorder = 2)
        ax.set_ylim([calibrated_chi2*.75, threshold_val * 1.2])
         
        #Plot threshold as dotted line
        x1 = [min(x_log), max(x_log)]
        y1 = [threshold_val, threshold_val]
        ax.plot(x1, y1, ':', color = 'dimgrey')
            
    #Save the figure
    plt.savefig('PROFILE LIKELIHOOD PLOTS.svg')

def plotPLConsequences(df, param_label):
    
    '''
    Purpose: Plot PPL results to evaluate consqeuences of unidentifiability 
            1) parameter relationships and 
            2) internal states
    
    Inputs: 
        df: a dataframe containing the results of the PPL (from Calc_PL) for the given parameter
        param_label: a string defining the identity of the parameter for which the PPL 
            consequences are to be plotted
        
    Outputs: None
    
    Figures:
        'PARAMETER RELATIONSHIPS ALONG ' + param_label + '.svg' 
            (plot of parameter relationships for parameter = param_label)
        'INTERNAL STATES ALONG ' + param_label + '.svg' 
            (plot of internal model state dynamics for parameter = param_label)
        
    '''
    #Define indices of free parameters
    indicies = []  
    for i, value in enumerate(p_all):
        label = real_param_labels_all[i]
        if label in real_param_labels_free:
            indicies.append(i)
    
    '''1. Plot parameter relationships'''
    fig = plt.figure(figsize = (3.5,4))
    
    x_ = list(df['fixed ' + param_label])
    x = [log10(val) for val in x_]
    y = list(df['fixed ' + param_label + ' all params'])
    
    lists = []
    for i in range(0, len(real_param_labels_all)):
        for j in range(0, len(real_param_labels_free)):
            if real_param_labels_all[i] == real_param_labels_free[j]:
                lists.append([log10(y_[i]) for y_ in y])
    
    sns.set_palette('mako')
    for j in range(0, len(real_param_labels_free)):
        plt.plot(x, lists[j], linestyle = 'dotted', marker = 'o', markerSize = 4, 
                 label = real_param_labels_free[j])
    
    plt.xlabel(param_label)
    plt.ylabel('other parameters')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
               fancybox=True, shadow=False, ncol=3)
   
    plt.savefig('PARAMETER RELATIONSHIPS ALONG ' + param_label + '.svg', dpi = 600)
    
    '''2. Plot internal model states'''
    n = len(y)
    color = plt.cm.Blues(np.linspace(.2, 1,n))
    plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    
    fig, axs = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=False, figsize = (8, 4))
    fig.subplots_adjust(hspace=.25)
    fig.subplots_adjust(wspace=0.2)
    
    for param_list in y: # for each parameter set
        t_1, t_2, solution_before, solution_on, state_labels = solveAll(param_list, exp_data)
        
        axs = axs.ravel()
        for i in range(0, len(state_labels)):
            axs[i].plot(t_1, solution_before[:,i], linestyle = 'dashed')
            axs[i].plot(t_2, solution_on[:,i])
            
            if i in [0, 4]:
                axs[i].set_ylabel('Simulation value (a.u.)', fontsize = 8)
            if i in [4,5,6,7]:
                axs[i].set_xlabel('Time (hours)', fontsize = 8)
            
            axs[i].set_title(state_labels[i], fontweight = 'bold', fontsize = 8)
            
        plt.savefig('INTERNAL STATES ALONG ' + param_label + '.svg', dpi = 600)
                   
def calcThresholdPL(calibrated_params, exp_data):
        print('CALIBRATED PARAMETERS - ORIGINAL EXPERIMENTAL DATA')  
        doses, norm_solutions_cal, chi2 = solveAll(calibrated_params, exp_data)
        calibrated_chi2 = chi2
        print('******************')
        print('chi2 CALIBRATED: ' + str(round(chi2, 4)))
    
        print('REFERENCE PARAMETERS - ORIGINAL EXPERIMENTAL DATA')  
        doses, norm_solutions_ref, chi2 = solveAll(p_ref, exp_data)
        print('chi2 REFERENCE: ' + str(round(chi2, 4)))
        print('******************')
        
        print('Calculating chi2 REFERENCE...')  
        
        #Generate noise realizations and calculate chi2_ref for each noise realization
        n_noise_realizations = 1000
        exp_data_original = exp_data
        exp_data_noise_list = []
        chi2_ref_list = []
    
        #Generate noise array
        #Define mean and standard error for error distribution
        mu = 0
        sigma =  .05 / sqrt(3)
        
        #Generate noise values
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
   
            #Re-normalize the data to the maximum value in the dataset
            if data == 'ligand dose response and DBD dose response':
                data_ligand = exp_data_noise[:11]
                data_DBD = exp_data_noise[11:]
                
                L_norm = [i/max(data_ligand) for i in data_ligand]
                DBD_norm = [i/max(data_DBD) for i in data_DBD]
                exp_data_noise_ = L_norm + DBD_norm
        
            else:
                exp_data_noise_ = [i/max(exp_data_noise) for i in exp_data_noise]
            
            data_dictionary['exp_data'] = exp_data_noise_
            exp_data = data_dictionary['exp_data']
            exp_data_noise_list.append(exp_data_noise_)
            
            #Calculate chi2_ref using noise realization i
            doses_ligand, norm_solutions, chi2_ref = solveAll(p_ref, exp_data)
            chi2_ref_list.append(chi2_ref)
          
        print('done')
        print('Calculating chi2 FIT...')   

        #Set up parallelization for calculating chi2_Fit by structuring parameters and 
        #noise realizations into a dataframe
        dfNoise = pd.DataFrame()
        dfNoise['count'] = range(0, n_noise_realizations)
        dfNoise['p'] = [calibrated_params] * n_noise_realizations
        dfNoise['fit labels'] = [fit_params] * n_noise_realizations
        dfNoise['exp data'] = exp_data_noise_list
        dfNoise['bounds'] = [bounds] * n_noise_realizations
    
        #Solve each parameter estimation problem with parallelization using the calibrated 
        #parameters as the initial guess
        df = dfNoise
        all_opt_results = []
        with mp.Pool(num_cores) as pool:
            result = pool.imap(optPar, df.itertuples(name=None))
            pool.close()
            pool.join()
            output = [[list(x[0]), list(x[1])] for x in result]
         
        #Unpack and restructure results
        for ig in range(0, len(output)):
            all_opt_results.append(output[ig][0])
            result_row_labels = output[ig][1]
            
        df_opt = pd.DataFrame(all_opt_results, columns = result_row_labels)
        chi2_fit = list(df_opt['chi2'])

        #Plot chi2 distribution and calculate 99% CI threshold
        x2 = []
        for i in range(0, len(chi2_ref_list)):
            val = chi2_ref_list[i] - chi2_fit[i]
            x2.append(val)
         
        #Define threshold value
        threshold_PL_val = np.percentile(x2, confidence_interval_label)
  
        #Plot results
        plt.figure(figsize = (5,3))
        plt.xlabel('chi2_ref - chi2_fit')
        plt.ylabel('Count')
        y, x, _ = plt.hist(x2, bins = 35, histtype='bar', color = 'dimgrey')
        plt.plot([threshold_PL_val, threshold_PL_val], [0, max(y)], '-', lw=3, 
                 alpha=0.6, color = sky_blue, linestyle = ':', 
                 label = str(confidence_interval_label) + '%')  
        plt.savefig('./CHI2 DISTRIBUTION', bbox_inches="tight", dpi = 600)
        
        print('******************')
        threshold_PL_val = np.round(threshold_PL_val, 1)
        print('CHI2 THRESHOLD FOR ' + str(num_vars) + ' PARAMETERS WITH ' + 
              str(confidence_interval_label) + '% CONFIDENCE')
        print(threshold_PL_val)
        print('******************')
        
        #Save results
        savePL(threshold_PL_val, calibrated_params, calibrated_chi2, real_param_labels_all)
        
        return threshold_PL_val
        
# =============================================================================
#  Module 1 - code to generate and simulate PEM EVALUATION data
# =============================================================================
def savePemEvalData(data_, data_type, filename, count):
    '''
    Purpose: Save PEM evaluation data in format usable by downstream code
    
    Inputs: 
        data_: a list of floats containing the normalized reporter expression values for each datapoint 
            (length = # datapoints)
        data_type: a string defining the type of data ('noise' or 'raw')
        filename: a string defining the filename to save the results to
        count: an integer defining the data set number that is being saved 
            (count = 0 -> (# PEM evaluation data_sets - 1))

    Outputs: 
        df_test: a dataframe containing the results
    
    Files:
        filename + data_type + '.xlsx' (df_test in Excel form)
        
    '''
   #Define the dataframe
    if data == 'ligand dose response only':
        df_test1 = pd.DataFrame(data_, columns = ['L'])
        df_test = pd.concat([df_test1], axis=1)
    
    elif data == 'ligand dose response and DBD dose response':
        df_test1 = pd.DataFrame(data_[:11], columns = ['L'])
        df_test2 = pd.DataFrame(data_[11:19], columns = ['DBD_20'])
        df_test3 = pd.DataFrame(data_[19:], columns = ['DBD_10'])
        df_test = pd.concat([df_test1, df_test2, df_test3], axis=1)
        
    #Save results
    filename = filename + data_type + '.xlsx'

    if count==1:
        with pd.ExcelWriter(filename) as writer:  # doctest: +SKIP
            df_test.to_excel(writer, sheet_name = data_type + ' ' + str(count))
    else: 
        path = filename
        book = load_workbook(path)
        writer = pd.ExcelWriter(path, engine = 'openpyxl')
        writer.book = book      
        df_test.to_excel(writer, sheet_name = data_type + ' ' + str(count))
        writer.save()
        writer.close() 
    return df_test
        
def generatePemEvalData(df_global_search, num_datasets):
    '''
    Purpose: Generate PEM evaluation data based on results of a global search
    
    Inputs: 
        dfGlobalSearch: a dataframe containing global search results
        numDatasets: an integer defining the number of PEM evaluation data_sets to generate

    Outputs: 
        df_list: a list of dataframes, each containing PEM evaluation data (with noise) 
            for a different dataset (length = number of PEM evaluation datasets)
    '''
    
    saveConditions(conditions_dictionary, initial_params_dictionary, data_dictionary)
    
    filtered_data, df_params  = filterGlobalSearch(df_global_search, num_datasets, 
                                                   all_param_labels, 'chi2')

    count = 1
    df_list = []
    Rsq_list = []
    for row in df_params.itertuples(name = None):
        #Define parameters
        p = [row[2], row[3], row[4], row[5], row[6], row[7]]
        doses_ligand, norm_solutions, chi2 = solveAll(p, exp_data)
        
        #Add noise
        noise_solutions = addNoise(norm_solutions, count)
        
        #Calculate R2 between PEM evaluation training data with and without noise
        Rsq = calcRsq(norm_solutions, noise_solutions)
        Rsq_list.append(Rsq)
        
        #Save results
        saveFilename = ''
        savePemEvalData(norm_solutions, ' PEM EVALUATION DATA RAW', saveFilename, count)
        df = savePemEvalData(noise_solutions, ' PEM EVALUATION DATA NOISE', saveFilename, count)
        df_list.append(df)
        count += 1
          
    return df_list, Rsq_list

def addNoise(raw_vals, count):
    '''
    Purpose: Add technical error to a dataset, according to a normal distribution 
        (defined by a mean and std)
    
    Inputs: 
        raw_vals: a list of floats defining the values before technical error is added 
        (length = # datapoints)
        count: the number corresponding to the given PEM evaluation dataset (count = 0 used 
               for reference data, counts = 1-8 used for PEM evaluation data)

    Outputs: 
        noise_vals_norm: a list of floats defining the values after technical error is added 
        (length = # datapoints)
    '''
    
    #Define mean and std for error distribution
    mu = 0
    sigma =  .05 / sqrt(3)
    
    #Generate noise values
    seeds = [3457, 1234, 2456, 7984, 7306, 3869, 5760, 9057, 2859]
    np.random.seed(seeds[count])
    noise = np.random.normal(mu, sigma, len(raw_vals))

    #Add noise to each data point
    noise_vals = []
    for i, noise_ in enumerate(noise):
        new_val = raw_vals[i] + noise_
        if new_val < 0.0:
            new_val = 0.0
        noise_vals.append(new_val)
    
    #Re-normalize the data to the maximum value in the dataset
    if data == 'ligand dose response and DBD dose response':
        data_ligand = noise_vals[:11]
        data_DBD = noise_vals[11:]
        
        L_norm = [i/max(data_ligand) for i in data_ligand]
        DBD_norm = [i/max(data_DBD) for i in data_DBD]
        noise_vals_norm = L_norm + DBD_norm

    else:
        noise_vals_norm = [i/max(noise_vals) for i in noise_vals]
        
    return noise_vals_norm


def unpackPemEvalData(output, df_results):
    '''
    Purpose: Unpack PEM evaluation results from global search and structure as df
    
    Inputs: 
        output: a list of lists defining the chi2 values for each PEM evaluation dataset, 
            for each parameter set (directly from multiprocessing code)
        df_results: a dataframe containing the original results of the global search
            
    Outputs: 
        df_results: a dataframe containing the restructured results of the global search
    '''
    
    chi2_1_list = []
    chi2_2_list = []
    chi2_3_list = []
    chi2_4_list = []
    chi2_5_list = []
    chi2_6_list = []
    chi2_7_list = []
    chi2_8_list = []

    for item in output:
        chi2_1_list.append(item[0])
        chi2_2_list.append(item[1])
        chi2_3_list.append(item[2])
        chi2_4_list.append(item[3])
        chi2_5_list.append(item[4])
        chi2_6_list.append(item[5])
        chi2_7_list.append(item[6])
        chi2_8_list.append(item[7])

    df_results['CF1'] = chi2_1_list
    df_results['CF2'] = chi2_2_list
    df_results['CF3'] = chi2_3_list
    df_results['CF4'] = chi2_4_list
    df_results['CF5'] = chi2_5_list
    df_results['CF6'] = chi2_6_list
    df_results['CF7'] = chi2_7_list
    df_results['CF8'] = chi2_8_list
    
    return df_results


def runGlobalSearchPemEval(n_search):
    '''
    Purpose: Run global search for PEM evaluation data
    
    Inputs: None
    
    Outputs:
        df_results: a dataframe containing the results of the global search
        
    Files:
        './GLOBAL SEARCH RESULTS.xlsx' (the dataframe df_results in Excel form)
    '''
    

    #Generate parameter sets
    df_params = generateParams(problem_free, n_search, p_all, problem_all_params, model)
    
    if data_type == 'PEM evaluation':
        if parallelization == 'no':  # without multiprocessing
            output = []
            for row in df_params.itertuples(name = None):
                result = solvePar(row)
                output.append(result)
        
        if parallelization == 'yes':  # with multiprocessing
            with mp.Pool(conditions_dictionary["num_cores"]) as pool:
                result = pool.imap(solvePar, df_params.itertuples(name = None))
                pool.close()
                pool.join()
                output = [[round(x[0],4),round(x[1],4), round(x[2],4), round(x[3],4), 
                           round(x[4],4), round(x[5],4), round(x[6],4), round(x[7],4)] 
                          for x in result]

        #Unpack GS results
        df_results = df_params
        df_results_unpacked = unpackPemEvalData(output, df_results)
        
    elif data_type == 'experimental': 
        if parallelization == 'no':  # without multiprocessing
            output = []
            for row in df_params.itertuples(name = None):
                result = solvePar(row)
                output.append(result)
        
        if parallelization == 'yes':  # with multiprocessing
            with mp.Pool(conditions_dictionary["num_cores"]) as pool:
                result = pool.imap(solvePar, df_params.itertuples(name = None))
                pool.close()
                pool.join()
                output = [[round(x[0],4)] for x in result]
                
        df_results = df_params
        df_results['chi2'] = output
        df_results_unpacked = df_results

    filename = './GLOBAL SEARCH RESULTS'
    with pd.ExcelWriter(filename + '.xlsx') as writer:  # doctest: +SKIP
        df_results_unpacked.to_excel(writer, sheet_name='')
    
    print('Global search complete')
    return df_results
  
    

def runOptPemEval(df_results, run):
    '''
    Purpose: Run filter and optimization for PEM evaluation data
    
    Inputs: 
        df_results: a dataframe containing the results of the global search
        run: a string defining the ID of the run (from 1 -> # PEM evaluation data_sets)
    
    Outputs:
        df_opt: a dataframe containing the results of the optimization (# rows = # initial guesses)
          
    Files:
        './GLOBAL SEARCH RESULTS.xlsx' (the dataframe df_results in Excel form)
        
    '''
    
    '''Filter'''
    #Filter resuts of global search
    sort_col = 'CF' + str(run)
    filtered_data, initial_guesses  = filterGlobalSearch(df_results, n_initial_guesses, 
                                                         all_param_labels, sort_col)
    
    '''Optimization'''
    #with multiprocessing
    df = initial_guesses
    df['exp_data'] = [exp_data] * len(df.index)
    df['bounds'] = [problem_free['bounds']] * len(df.index)
    all_opt_results = []
    
    if parallelization == 'no':   # without multiprocessing
        for row in df.itertuples(name = None):
            result_row, result_row_labels = optPar(row)
            all_opt_results.append(result_row)
            
    if parallelization == 'yes':   # with multiprocessing 
        with mp.Pool(num_cores) as pool:
            result = pool.imap(optPar, df.itertuples(name=None))
            pool.close()
            pool.join()
            output = [[list(x[0]), list(x[1])] for x in result]
        for ig in range(0, len(output)):
            all_opt_results.append(output[ig][0])
            result_row_labels = output[ig][1]
        
    df_opt = pd.DataFrame(all_opt_results, columns = result_row_labels)

    #Sort by chi2
    df_opt = df_opt.sort_values(by=['chi2'], ascending = True)
    
    #Calculate Rsq and add to df
    Rsq_list = []
    for j in range(0, n_initial_guesses):
        params = []
        for i in range(0, len(p_all)):
            col_name = real_param_labels_all[i] + '*'
            val = df_opt[col_name].iloc[j]
            params.append(val)
        doses, norm_solutions, chi2 = solveAll(params, exp_data)
        Rsq = calcRsq(norm_solutions, exp_data)  
        Rsq_list.append(Rsq)

    df_opt['Rsq'] = Rsq_list
    
    #Save df
    filename = './OPT RESULTS'
    with pd.ExcelWriter(filename + '.xlsx') as writer:  # doctest: +SKIP
        df_opt.to_excel(writer, sheet_name='')
   
    #Plot results
    plotTrainingDataFits(df_opt)
    
    return df_opt


def defineExpPemEval(df, data):
    '''
    Purpose: Define data for PEM evaluation problems and strucutre in way that is 
        compatible with downstream code
    
    Inputs: 
        df: a dataframe containing the PEM evaluation data
        data: a string defining the data identity
    
    Outputs:
        exp_data: a list of floats containing the PEM evaluation data (length = # datapoints)
          
    '''
    
    data_ligand = list(df['L'])
    if data == 'ligand dose response only':
        exp_data = data_ligand
    elif data == 'ligand dose response and DBD dose response':
        data_dbd_20 = list(df['DBD_20'])[:8]
        data_dbd_10 = list(df['DBD_10'])[:8]
        exp_data = data_ligand + data_dbd_20 + data_dbd_10
 
    return exp_data

# =============================================================================
# Code to RUN the workflow
# =============================================================================

#Record start time
startTime_0 = datetime.datetime.now()
      
if 1 in modules:
    module = 1
    module_color = black_
    
    #Record start time
    startTime_1 = datetime.datetime.now()
        
    #STEP 1: generate PEM evaluation data by running a global search with respect to the experimental/training data 
    #Set up file structure
    os.chdir(full_path)
    sub_folder_name = 'MODULE 1 - PARAMETER ESTIMATION WITH PEM EVALUATION DATA'
    createFolder('./' + sub_folder_name)
    os.chdir('./' + sub_folder_name)
    sub_folder_name = 'GENERATE PEM EVALUATION DATA'
    createFolder('./' + sub_folder_name)
    os.chdir('./' + sub_folder_name)
    
    print('Generating PEM evaluation data...')
    exp_data_original = data_dictionary["exp_data"]
  
    #Set data_type to experimental
    data_dictionary["data_type"] = 'experimental'
    data_type = data_dictionary["data_type"]
    n_initial_guesses = num_datasets_pem_eval
    
    #Run the global search and output the df of filtered results
    PL_ID = 'no'
    df_results = runGlobalSearchPemEval(n_search_pem_eval)
    chi2_vals = list(df_results['chi2'])
    
    #Restructure chi2 values
    vals = []
    for item in chi2_vals:
        vals.append(item[0])
  
    #Calculate 10th percentile of chi2 values across global search
    chi2_10_percent = np.percentile(vals, 10)
  
    #Generate PEM evaluation data (add noise and save)
    df_pem_eval, Rsq_list = generatePemEvalData(df_results, num_datasets_pem_eval)
    R_sq_pem_eval_pass = np.round(np.mean(Rsq_list), 3)
    print('R2pass: '+ str(R_sq_pem_eval_pass))
  
    os.chdir('../')
    
    #STEP 2: Run parameter estimation with PEM evaluation data
    print('Running parameter estimation with PEM evaluation data...')
    #Set up file structure
    sub_folder_name = 'PARAMETER ESTIMATION WITH PEM EVALUATION DATA'
    createFolder(sub_folder_name)
    os.chdir('./' + sub_folder_name)
    
    #Set data_type to PEM evaluation
    data_dictionary["data_type"] = 'PEM evaluation'
    data_type = data_dictionary["data_type"]
    
    #Change n_search back to the value used for parameter estimation (set in Settings.py)
    conditions_dictionary["n_search"] = n_search_parameter_estimation
    n_search = conditions_dictionary["n_search"]
    n_initial_guesses = conditions_dictionary["n_initial_guesses"]
 
    #Run parameter estimation workflow for each PEM evaluation data set
    data_sets = np.arange(0, len(df_pem_eval))
    numDatasets = len(data_sets)
    files = []
    for i in range(0, len(data_sets)): #for each PEM evaluation data set
        #Update run (i starts at 0, so the first run should be i + 1)
        run = str(i + 1)
        conditions_dictionary['run'] = run
        
        #Define PEM evaluation data for this run
        data_dictionary["exp_data"] = defineExpPemEval(df_pem_eval[i], data)
        exp_data = data_dictionary["exp_data"]
        data_dictionary["error"] = []

        print('Starting optimization for dataset number ' + str(run) + '...')
        
        #Run the global search (only once - can calculate all 8 cost functions with the same
        #simulation results)
        if i == 0:
            df_results = runGlobalSearchPemEval(n_search)
              
        #Make subfolder, change to appropriate directory, and save conditions
        sub_folder_name = 'RUN ' + run
        createFolder(sub_folder_name)
        os.chdir('./' + sub_folder_name)
        saveConditions(conditions_dictionary, initial_params_dictionary, data_dictionary)
        
        #Run optimization module and add the output df to the list of output files
        df_opt = runOptPemEval(df_results, run) 
        files.append(df_opt)
        
        #Go up one directory
        os.chdir('../')
        
    #Run analysis on PEM evaluation data
    plotPemEvaluation(files, R_sq_pem_eval_pass, chi2_10_percent)
    
    #Set exp data back to the original training dataset
    exp_data = exp_data_original
    data_dictionary["exp_data"] = exp_data
    
    #Print stop time
    stopTime_1 = datetime.datetime.now()
    elapsed_time = stopTime_1 - startTime_1
    elapsed_time_total = round(elapsed_time.total_seconds(), 1)
    elapsed_time_total = elapsed_time_total/60
    print('')
    print('********************************************')
    print('MODULE 1')
    print('Total run time (min): ' + str(round(elapsed_time_total, 3)))
    print('Total run time (hours): ' + str(round(elapsed_time_total/60, 3)))   
    print('********************************************')
    
if 2 in modules:
    module = 2
    module_color = black_
    
    run = '1'
    conditions_dictionary['run'] = run

    ##Record start time
    startTime_2 = datetime.datetime.now()

    #Set data_type to experimental 
    data_dictionary["data_type"] = 'experimental'
    data_type = data_dictionary["data_type"]
     
    #Set up file structure and save conditions
    os.chdir(full_path)
    sub_folder_name = 'MODULE 2 - FIT TO EXPERIMENTAL DATA'
    createFolder('./' + sub_folder_name)
    os.chdir('./' + sub_folder_name)
    saveConditions(conditions_dictionary, initial_params_dictionary, data_dictionary)
    
    #Run parameter estimation method 
    PL_ID = 'no'
    df_opt, calibrated_params = runParameterEstimation()
    
    #Print stop time
    stopTime_2 = datetime.datetime.now()
    elapsed_time = stopTime_2 - startTime_2
    elapsed_time_total = round(elapsed_time.total_seconds(), 1)
    elapsed_time_total = elapsed_time_total/60
    print('')
    print('********************************************')
    print('MODULE 2')
    print('Total run time (min): ' + str(round(elapsed_time_total, 3)))
    print('Total run time (hours): ' + str(round(elapsed_time_total/60, 3))) 
    print('********************************************')

    
if 3 in modules:
    module = 3
    module_color = black_
    startTime_3 = datetime.datetime.now()
    PL_ID = 'yes'
    
    os.chdir(full_path)
    sub_folder_name = 'MODULE 3 - PARAMETER IDENTIFIABILITY ANALYSIS'
    createFolder('./' + sub_folder_name)
    os.chdir('./' + sub_folder_name)
    
    # =============================================================================
    # DEtermine the chi2 threshold
    # =============================================================================
    confidence_interval_label = confidence_interval*100
    #set the threshold by plotting the chi2 distribution and finding the CI
    threshold_PL_val = calcThresholdPL(calibrated_params, exp_data)  
   
    # =============================================================================
    # Calculate profile likelihood
    # =============================================================================
    n_search = conditions_dictionary["n_search"]
    n_initial_guesses = conditions_dictionary["n_initial_guesses"]
    exp_data = exp_data_original
    data_dictionary["exp_data"] = exp_data_original
    
    #Calculate profile likelihood for each free parameter
    PL_ID = 'yes' 
    doses_ligand, norm_solutions_cal, calibrated_chi2 = solveAll(calibrated_params, exp_data)
    df_list = calcPL(calibrated_params, calibrated_chi2, threshold_PL_val)
    
    #Plot internal model states and parameter relationships based on the 
    #results of the profile likelihood 
    save_internal_states_flag = True
    for i in range(0, len(real_param_labels_free)):
        plotPLConsequences(df_list[i], real_param_labels_free[i])    
    
    #Print stop time
    stopTime_3 = datetime.datetime.now()
    elapsed_time = stopTime_3 - startTime_3
    elapsed_time_total = round(elapsed_time.total_seconds(), 1)
    elapsed_time_total = elapsed_time_total/60
    print('')
    print('********************************************')
    print('MODULE 3')
    print('Total run time (min): ' + str(round(elapsed_time_total, 3)))
    print('Total run time (hours): ' + str(round(elapsed_time_total/60, 3)))    
    print('********************************************')
    
stop_time_all = datetime.datetime.now()
elapsed_time = stop_time_all - startTime_0
elapsed_time_total = round(elapsed_time.total_seconds(), 1)
elapsed_time_total = elapsed_time_total/60
print('')
print('********************************************')
print('MODULES: ' + str(modules))
print('Total run time (min): ' + str(round(elapsed_time_total, 3)))
print('Total run time (hours): ' + str(round(elapsed_time_total/60, 3))) 
print('********************************************')



    
