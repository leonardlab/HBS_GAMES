#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:41:51 2022

@author: kate
"""
import numpy as np
import os
from typing import Tuple
from config import Settings, ExperimentalData, Context, save_conditions, create_folder
from analysis import plot_x_y
from set_model import model
from global_search import generate_parameter_sets, solve_global_search
from optimization import Optimization
from solve_single import solve_single_parameter_set
from parameter_estimation_method_evaluation import generate_pem_evaluation_data, define_initial_guesses_for_pem_eval, optimize_pem_evaluation_data, plot_pem_evaluation
from parameter_profile_likelihood import calculate_threshold_chi_sq, calculate_profile_likelihood

class Modules:
    """Class with methods to run each module from the GAMES workflow"""

    @staticmethod
    def run_single_parameter_set() -> Tuple[list, float, float]:
        """Solves model for a single parameter set using dataID defined in Settings.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        sub_folder_name = "TEST SINGLE PARAMETER SET"
        path = create_folder(sub_folder_name)
        os.chdir(path)
        save_conditions()
        model.parameters = Settings.parameters
        solutions_norm, chi_sq, r_sq = solve_single_parameter_set()
        filename = "FIT TO TRAINING DATA"
        plot_x_y(
            [
                ExperimentalData.x,
                solutions_norm,
                ExperimentalData.exp_data,
                ExperimentalData.exp_error,
            ],
            [ExperimentalData.x_label, ExperimentalData.y_label],
            filename,
            ExperimentalData.x_scale,
        )
        
        print("R_sq = " + str(np.round(r_sq, 4)))
        print("chi_sq = " + str(np.round(chi_sq, 4)))
        print("*************************")
        
        return solutions_norm, chi_sq, r_sq

    def estimate_parameters() -> None:
        """Runs parameter estimation method (multi-start optimization)

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        sub_folder_name = "MODULE 2 - FIT TO EXPERIMENTAL DATA"
        path = create_folder(sub_folder_name)
        os.chdir(path)
        save_conditions()

        print("Starting global search...")
        ExperimentalData.data_type = 'training'
        df_parameters = generate_parameter_sets(Settings.parameter_estimation_problem_definition)
        df_global_search_results = solve_global_search(df_parameters)
        print("Global search complete.")

        print("Starting optimization...")
        _, calibrated_chi_sq, _, calibrated_parameters = Optimization.optimize_all(df_global_search_results)
        
        return calibrated_chi_sq, calibrated_parameters

  
    @staticmethod
    def evaluate_parameter_estimation_method() -> None:
        """Runs parameter estimation method evaluation by first generating 
        PEM evaluation data and then running multi-start optimziation with each set of PEM evaluation data 

        Parameters
        ----------
        None

        Returns
        -------
        None
    
        """
        sub_folder_name = "MODULE 1 - EVALUATE PARAMETER ESTIMATION METHOD"
        path = create_folder(sub_folder_name)
        os.chdir(path)
        save_conditions()
        
        print("Generating PEM evaluation data...")
        df_parameters = generate_parameter_sets(Settings.parameter_estimation_problem_definition)
        df_global_search_results = solve_global_search(df_parameters)
        pem_evaluation_data_list, chi_sq_pem_evaluation_criterion = generate_pem_evaluation_data(df_global_search_results)
        print("PEM evaluation data generated.")
        
        print("Starting optimization for PEM evaluation data...")
        ExperimentalData.data_type = 'PEM evaluation'
        df_initial_guesses_list = define_initial_guesses_for_pem_eval(df_global_search_results, pem_evaluation_data_list)
        df_list = optimize_pem_evaluation_data(df_initial_guesses_list, pem_evaluation_data_list, chi_sq_pem_evaluation_criterion)
        plot_pem_evaluation(df_list, chi_sq_pem_evaluation_criterion)
        print('PEM evaluation complete')
        

    def calculate_parameter_profile_likelihood(calibrated_chi_sq: float, calibrated_parameters: list) -> None:
        """Calculates parameter profile likelihood
        
        Parameters
        ----------
        calibrated_chi_sq
            a float defining the chi_sq associated with the calibrated parameter set
   
        calibrated_parameters
            a list of floats containing the calibrated values for each parameter
   
        Returns
        -------
        None
        """
        sub_folder_name = "MODULE 3 - PARAMETER IDENTIFIABILITY ANALYSIS"
        path = create_folder(sub_folder_name)
        os.chdir(path)
        save_conditions()
        
        threshold_chi_sq = calculate_threshold_chi_sq(calibrated_parameters, calibrated_chi_sq)
        # for parameter_label in Settings.parameter_labels:
        #     df = calculate_profile_likelihood(parameter_label, self.calibrated_parameters, self.calibrated_chi_sq)
        
        
    
