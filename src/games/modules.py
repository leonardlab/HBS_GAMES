#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:41:51 2022

@author: kate
"""
import os
from config import Settings, ExperimentalData,  Save
from analysis import Plots
from set_model import model
from global_search import GlobalSearch
from optimization import Optimization
from test_single import TestSolveSingle


class Modules:
    
    def test_single_parameter_set():
        """Solves model for a single parameter set using dataID defined in Settings.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None 
        
        """
        sub_folder_name = 'TEST SINGLE PARAMETER SET'
        path = Save.createFolder(sub_folder_name)
        os.chdir(path)
        Save.saveConditions()
        model.parameters = Settings.parameters
        solutions_norm, chi_sq, r_sq = TestSolveSingle.solve_single_parameter_set()
        filename = 'FIT TO TRAINING DATA'
        Plots.plot_x_y(ExperimentalData.x, solutions_norm, ExperimentalData.exp_data, ExperimentalData.exp_error, ExperimentalData.x_label, ExperimentalData.y_label, filename, ExperimentalData.x_scale)
        return solutions_norm, chi_sq, r_sq
    
    def estimate_parameters():
        """Runs parameter estimation method (multi-start optimization)
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None 
        
        """
        sub_folder_name = 'MODULE 2 - FIT TO EXPERIMENTAL DATA'
        path = Save.createFolder(sub_folder_name)
        os.chdir(path)
        Save.saveConditions()
        
        print('Starting global search...')
        df_parameters = GlobalSearch.generate_parameter_sets(Settings.parameter_estimation_problem_definition)
        df_global_search_results = GlobalSearch.solve_global_search(df_parameters)
        print('Global search complete.')
        
        print('Starting optimization...')
        Optimization.optimize_all(df_global_search_results)
        
    def generate_parameter_estimation_method_evaluation_data():
        pass
    
    def evaluate_parameter_estimation_method():
        pass
    
    def calculate_profile_likelihood():
        pass

