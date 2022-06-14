#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:38:59 2022

@author: kate
"""
from modules.solve_single import Solve_single
from modules.parameter_estimation import estimate_parameters
from modules.parameter_estimation_method_evaluation import evaluate_parameter_estimation_method
from modules.parameter_profile_likelihood import calculate_parameter_profile_likelihood

# =============================================================================
# Solve model for single set of parameters
# =============================================================================
#solutions_norm, chi_sq, r_sq = Solve_single.plot_single_parameter_set()

# =============================================================================
# Module 1 - evaluate PEM
# =============================================================================
evaluate_parameter_estimation_method()

# =============================================================================
# Module 2 - estimate parameters
# =============================================================================
#calibrated_chi_sq, calibrated_parameters = estimate_parameters()

# =============================================================================
# Module 3 - calculate PPL (must run 2 and 3 together or provide calibrated_chi_sq and calibrated_parameters)
# =============================================================================
#calculate_parameter_profile_likelihood(calibrated_chi_sq, calibrated_parameters)
