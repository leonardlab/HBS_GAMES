#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:49:45 2022

@author: kate
"""
import os
from utilities.saving import create_folder
from modules.parameter_estimation.global_search import generate_parameter_sets, solve_global_search
from config.settings import settings, parameter_estimation_problem_definition, folder_path
from config.experimental_data import ExperimentalData
from plots.plots_pem_evaluation import plot_pem_evaluation
from modules.parameter_estimation_method_evaluation.generate_pem_evaluation_data import generate_pem_evaluation_data
from modules.parameter_estimation_method_evaluation.evaluate_parameter_estimation_method import define_initial_guesses_for_pem_eval, optimize_pem_evaluation_data
 
def run_parameter_estimation_method_evaluation() -> None:
    """Runs parameter estimation method evaluation by first generating
    PEM evaluation data and then running multi-start optimziation 
    with each set of PEM evaluation data

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    sub_folder_name = "MODULE 1 - EVALUATE PARAMETER ESTIMATION METHOD"
    path = create_folder(folder_path, sub_folder_name)
    os.chdir(path)

    print("Generating PEM evaluation data...")
    df_parameters = generate_parameter_sets(parameter_estimation_problem_definition)
    df_global_search_results = solve_global_search(df_parameters, run_type = 'PEM evaluation')
    pem_evaluation_data_list, chi_sq_pem_evaluation_criterion = generate_pem_evaluation_data(
        df_global_search_results
    )
    print("PEM evaluation data generated.")

    print("Starting optimization for PEM evaluation data...")
    ExperimentalData.data_type = "PEM evaluation"
    df_initial_guesses_list = define_initial_guesses_for_pem_eval(
        df_global_search_results, pem_evaluation_data_list
    )
    df_list = optimize_pem_evaluation_data(
        df_initial_guesses_list, pem_evaluation_data_list, chi_sq_pem_evaluation_criterion
    )
    plot_pem_evaluation(df_list, chi_sq_pem_evaluation_criterion)
    print("PEM evaluation complete")
    



