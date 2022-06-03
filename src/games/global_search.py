#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 15:25:19 2022

@author: kate
"""

import numpy as np
import pandas as pd
from SALib.sample import latin
from set_model import model
from test_single import TestSolveSingle
from config import Settings

class GlobalSearch:
    def generate_parameter_sets(problem):
        """
        Generate parameter sets for global search 
        
        Parameters
        ----------
        problem
            a dictionary including the number, labels, and bounds for the free parameters 
     
            
        Returns
        -------
        df_parameters
            df with columns defining to parameter identities  and rows defining the paramter values for each set in the sweep
        """
        
      
        fit_params = problem['names'] #only the currently free parameters (as set in settings)
        num_params = problem['num_vars']
        all_params = Settings.parameter_labels #all params that are potentially free 
        n_search = Settings.num_parameter_sets_global_search
        
        #Create an empty dataframe to store results of the parameter sweep
        df_parameters = pd.DataFrame()
        
        #Fill each column of the dataframe with the intial values set in Settings. 
        for item in range(0, len(all_params)):
            param = all_params[item]
            param_array = np.full((1, n_search), Settings.parameters[item])
            param_array = param_array.tolist()
            df_parameters[param] = param_array[0]
    
        #Perform LHS
        param_values = latin.sample(problem, n_search, seed=456767)
    
        #Transform back to to linear scale
        params_converted = []
        for item in param_values:
            params_converted.append([10**(val) for val in item])
        params_converted = np.asarray(params_converted)
            
        #Replace the column of each fit parameter with the list of parameters from the sweep
        for item in range(0, num_params):   
            for name in fit_params:
                if fit_params[item] == name:
                    df_parameters[name] = params_converted[:,item]
       
        with pd.ExcelWriter('./PARAMETER SWEEP.xlsx') as writer:  
            df_parameters.to_excel(writer, sheet_name='GS parameters')
        
        return df_parameters
    
    def solve_global_search(df_parameters):
        """
        Generate parameter sets for global search 
        
        Parameters
        ----------
        df_parameters
            df with columns defining to parameter identities  and rows defining the paramter values for each set in the sweep
     
            
        Returns
        -------
        df_global_search_results
            df that contains the information in df_parameters, along with an extra column defining the cost function for each parameter set
      
        """

        chi_sq_list = []
        for row in df_parameters.itertuples(name = None):
            model.parameters = list(row[1:])
            solutions_norm, chi_sq, r_sq = TestSolveSingle.solve_single_parameter_set()
            chi_sq_list.append(chi_sq)
        
        df_global_search_results = df_parameters
        df_global_search_results['chi_sq'] = chi_sq_list
        
        with pd.ExcelWriter('GLOBAL SEARCH RESULTS.xlsx') as writer: 
             df_global_search_results.to_excel(writer, sheet_name='GS results')
        
        return df_global_search_results