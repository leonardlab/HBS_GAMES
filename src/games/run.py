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
#solutions_norm, chi_sq, r_sq = Modules.test_single_parameter_set()   

# =============================================================================
# Module 2 - estimate parameters
# =============================================================================
Modules.estimate_parameters()