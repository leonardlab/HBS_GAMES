#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 15:25:37 2022

@author: kate
"""

import numpy as np
import pandas as pd
from lmfit import Model as Model_lmfit
from lmfit import Parameters as Parameters_lmfit

from set_model import model
from test_single import TestSolveSingle
from config import Settings, ExperimentalData
from analysis import Plots

class Optimization():
    
    def generalize_parameter_labels():
        """Generates generalized parameter labels (p1, p2, etc.) for use in optimization functions
        
        Parameters
        ----------
        None
            
        Returns
        -------
        general_parameter_labels
            list of floats defining the generalized parameter labels 
        
        general_parameter_labels_free 
            list of floats defining which of the generalized parameter labels are free
        
        """
        general_parameter_labels = ['p' + str(i+1) for i in range(0, len(Settings.parameters))]
        general_parameter_labels_free = ['p' + str(i+1) for i in Settings.free_parameter_indicies]
        return general_parameter_labels, general_parameter_labels_free 
    
    def define_best_optimization_results(df_optimization_results):
        """Prints best parameter set from optimization to console and plots the best fit to training data
        
        Parameters
        ----------
        df_optimization_results
            df containing the results of all optimization runs
            
        Returns
        -------
        None
        
        """
        
        df = df_optimization_results
        best_case_parameters = []
        for i in range(0, len(Settings.parameters)):
            col_name = Settings.parameter_labels[i] + '*'
            val = df[col_name].iloc[0]
            best_case_parameters.append(val)
        best_case_parameters = [np.round(parameter, 5) for parameter in best_case_parameters]
        solutions_norm = df['Simulation results'].iloc[0]
        filename = 'BEST FIT TO TRAINING DATA'
        Plots.plot_x_y(ExperimentalData.x, solutions_norm, ExperimentalData.exp_data, ExperimentalData.exp_error, ExperimentalData.x_label, ExperimentalData.y_label, filename, ExperimentalData.x_scale)
   
        filename = 'COST FUNCTION TRAJECTORY'
        y = list(df['chi_sq_list'].iloc[0])
        Plots.plot_x_y(range(0, len(y)), y, 'None', 'None', 'function evaluation', 'chi_sq', filename)
   
        print('*************************')
        print('Calibrated parameters: ')
        for i, label in enumerate(Settings.parameter_labels):
            print(label + ' = ' + str(best_case_parameters[i]))
        print('')
        
        r_sq_opt = round(df['r_sq'].iloc[0], 3)
        chi_sq_opt_min = round(df['chi_sq'].iloc[0], 3)
        
        print('Metrics:')
        print('R_sq = ' + str(r_sq_opt))
        print('chi_sq = ' + str(chi_sq_opt_min))
        print('*************************')
   
 
    def optimize_all(df_global_search_results):
        """Runs optimization for each intitial guess and saves results
        
        Parameters
        ----------
        df_global_search_results
            df containing the results of the global search
            
        Returns
        -------
        None
        
        """
        df = df_global_search_results.sort_values(by=['chi_sq'])
      
        results_row_list = []
        for i, row in enumerate(df.itertuples(name = None)):
            if i < Settings.num_parameter_sets_optimization:
                initial_parameters = list(row[1:len(Settings.parameters) + 1])
                results_row, results_row_labels = Optimization.optimize_single_initial_guess(initial_parameters, i)
                results_row_list.append(results_row)
        
        print('Optimization complete.')
        df_optimization_results = pd.DataFrame(results_row_list, columns = results_row_labels)
        df_optimization_results = df_optimization_results.sort_values(by=['chi_sq'], ascending = True)
        Optimization.define_best_optimization_results(df_optimization_results)
        
        with pd.ExcelWriter('./OPTIMIZATION RESULTS.xlsx') as writer:  
            df_optimization_results.to_excel(writer, sheet_name='opt')

            
    def define_parameters_for_opt(initial_parameters):
        """Defines parameters for optimization with structure necessary for LMFit optimization code
        
        Parameters
        ----------
        initial_parameters
            a list of floats containing the initial guesses for each parameter 
            
        Returns
        -------
        params_for_opt
            an object defining the parameters and bounds for optimization (in structure necessary for LMFit optimization code)
        
        """
        general_parameter_labels, general_parameter_labels_free = Optimization.generalize_parameter_labels()
        
        #Set default values
        bounds = Settings.parameter_estimation_problem_definition['bounds']
        num_parameters = len(Settings.parameters)
        bound_min_list = [0] * num_parameters
        bound_max_list = [np.inf] * num_parameters
        vary_list = [False] * num_parameters
    
        #Set min and max bounds and vary_index by comparing free parameters lists with list of all parameters
        for param_index in range(0, len(general_parameter_labels)): #for each parameter
            for free_param_index in range(0, len(general_parameter_labels_free)): #for each free parameter
            
                #if param is free param, change vary to True and update bounds
                if general_parameter_labels[param_index] == general_parameter_labels_free[free_param_index]: 
                    vary_list[param_index] = True
                    bound_min_list[param_index] = 10 ** bounds[free_param_index][0]
                    bound_max_list[param_index] = 10 ** bounds[free_param_index][1]          
       
        #Add parameters to the parameters class
        params_for_opt = Parameters_lmfit()
        for index_param in range(0, len(general_parameter_labels)):
            params_for_opt.add(general_parameter_labels[index_param], value=initial_parameters[index_param], 
                        vary = vary_list[index_param], min = bound_min_list[index_param], 
                        max = bound_max_list[index_param])
            
        for null_index in range(len(general_parameter_labels), 10):
            params_for_opt.add('p' + str(null_index + 1), value=0, 
                        vary = False, min = 0, 
                        max = 1)
    
        return params_for_opt
    
    def define_results_row(initial_parameters, results, chi_sq_list):
        """Defines results for each optimization run
        
        Parameters
        ----------
        initial_parameters
            a list of floats containing the initial guesses for each parameter 
            
        results
            an object containing results of the given optimization run
            
        chi_sq_list
            a list of floats containing the chi_sq values for each function evaluation
            
        Returns
        -------
        results_row
            a list of floats and lists containing the results for the given optimization run
            
        results_row_labels 
            a list of strings defining the labels for each item in results_row
        
        """
        #add initial parameters to row for saving 
        results_row = []
        results_row_labels = []
        for i, val in enumerate(initial_parameters):
            results_row.append(val)
            results_row_labels.append(Settings.parameter_labels[i])
        
        #Define best fit parameters (results of optimization run)   
        best_fit_params = list(results.params.valuesdict().values())[:len(initial_parameters)]
      
        #Solve ODEs with final optimized parameters, calculate chi2, and add to result_row for saving
        model.parameters = best_fit_params
        solutions_norm, chi_sq, r_sq = TestSolveSingle.solve_single_parameter_set()
        
        results_row.append(chi_sq)
        results_row_labels.append('chi_sq')
        
        results_row.append(r_sq)
        results_row_labels.append('r_sq')
        
        #append best fit parameters to results_row for saving (with an * at the end of each parameter name
        #to denote that these are the optimized values)
        for index in range(0, len(best_fit_params)):
            results_row.append(best_fit_params[index])
            label = Settings.parameter_labels[index] + '*'
            results_row_labels.append(label)
        
        #Define other conditions and result metrics and add to result_row for saving
        #Note that results.redchi is the chi2 value directly from LMFit and should match the chi2
        #calculated in this code if the same cost function as LMFit is used
        items = [results.redchi, results.success, model, 
                  chi_sq_list, solutions_norm]
        item_labels = ['redchi2',  'success', 'model', 'chi_sq_list', 
                        'Simulation results']
      
        for i in range(0, len(items)):
            results_row.append(items[i])
            results_row_labels.append(item_labels[i])
      
        return results_row, results_row_labels 

    def optimize_single_initial_guess(initial_parameters, i): 
        """Runs optimization for a single initial guess
        
        Parameters
        ----------
        initial_parameters
            a list of floats containing the initial guesses for each parameter 
            
        i
            an integer defining the optimization run number 
            
        Returns
        -------
        results_row
            a list of floats and lists containing the results for the given optimization run
            
        results_row_labels 
            a list of strings defining the labels for each item in results_row """
            
        count = i + 1
        chi_sq_list = []
        
        def solve_for_opt(x, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0, p7 = 0, p8 = 0, p9 = 0, p10 = 0):
            p = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
            model.parameters = p[:len(Settings.parameters)]
            solutions_norm, chi_sq, r_sq = TestSolveSingle.solve_single_parameter_set()
            chi_sq_list.append(chi_sq)
            return np.array(solutions_norm)
            
        params_for_opt = Optimization.define_parameters_for_opt(initial_parameters)
        model_ = Model_lmfit(solve_for_opt, nan_policy='propagate')
        
        if Settings.weight_by_error == 'no':
            weights_ = [1] * len(ExperimentalData.exp_error)
        else:
            weights_ = [1/i for i in ExperimentalData.exp_error] 
        results = model_.fit(ExperimentalData.exp_data, params_for_opt, method = 'leastsq', x = ExperimentalData.x, weights = weights_)
        print('Optimization round ' + str(count) + ' complete.')
        
        results_row, results_row_labels = Optimization.define_results_row(initial_parameters, results, chi_sq_list)
     
        return results_row, results_row_labels
    