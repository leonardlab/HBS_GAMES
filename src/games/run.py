#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:38:59 2022

@author: kate
"""

from modules import Modules

# =============================================================================
# Solve model for single set of parameters
# =============================================================================
#solutions_norm, chi_sq, r_sq = Modules.run_single_parameter_set()

# =============================================================================
# Module 1 - evaluate PEM
# =============================================================================
#Modules.evaluate_parameter_estimation_method()

# =============================================================================
# Module 2 - estimate parameters
# =============================================================================
calibrated_chi_sq, calibrated_parameters = Modules.estimate_parameters()

# =============================================================================
# Module 3 - calculate PPL (must run 2 and 3 together for 3 to run)
# =============================================================================
Modules.calculate_parameter_profile_likelihood(calibrated_chi_sq, calibrated_parameters)
