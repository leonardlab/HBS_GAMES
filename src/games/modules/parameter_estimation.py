#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:39:44 2022

@author: kate
"""
import os
from config import Settings, ExperimentalData
from utilities.saving import save_conditions, create_folder
from modules.optimization import Optimization
from modules.global_search import generate_parameter_sets, solve_global_search

def estimate_parameters() -> None:
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
   path = create_folder(sub_folder_name)
   os.chdir(path)
   save_conditions()

   print("Starting global search...")
   ExperimentalData.data_type = "training"
   df_parameters = generate_parameter_sets(Settings.parameter_estimation_problem_definition)
   df_global_search_results = solve_global_search(df_parameters)
   print("Global search complete.")

   print("Starting optimization...")
   _, calibrated_chi_sq, _, calibrated_parameters = Optimization.optimize_all(
       df_global_search_results
   )

   return calibrated_chi_sq, calibrated_parameters
   