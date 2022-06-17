#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:39:44 2022

@author: kate
"""
import os
from config.settings import settings, parameter_estimation_problem_definition, folder_path
from config.experimental_data import ExperimentalData
from utilities.saving import create_folder
from modules.parameter_estimation.optimization import optimize_all
from modules.parameter_estimation.global_search import generate_parameter_sets, solve_global_search

def run_parameter_estimation() -> None:
   """Runs parameter estimation method (multi-start optimization)

   Parameters
   ----------
   None

   Returns
   -------
   calibrated_chi_sq
       a float defining the chi_sq associated with the calibrated parameter set

   calibrated_parameters
       a list of floats containing the calibrated values for each parameter

   """
   sub_folder_name = "MODULE 2 - FIT TO EXPERIMENTAL DATA"
   path = create_folder(folder_path, sub_folder_name)
   os.chdir(path)

   print("Starting global search...")
   ExperimentalData.data_type = "training"
   df_parameters = generate_parameter_sets(parameter_estimation_problem_definition)
   df_global_search_results = solve_global_search(df_parameters)
   print("Global search complete.")

   print("Starting optimization...")
   _, calibrated_chi_sq, _, calibrated_parameters = optimize_all(
       df_global_search_results
   )

   return calibrated_chi_sq, calibrated_parameters
   