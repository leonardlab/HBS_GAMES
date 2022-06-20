#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:24:39 2022

@author: kate
"""
import os
import numpy as np
from config.settings import settings, folder_path
from utilities.saving import create_folder
from modules.parameter_profile_likelihood.calculate_parameter_profile_likelihood import calculate_ppl
from modules.parameter_profile_likelihood.calculate_threshold import calculate_threshold_chi_sq

def run_parameter_profile_likelihood(
        calibrated_chi_sq: float, calibrated_parameters: list
    ) -> None:
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
    path = create_folder(folder_path, sub_folder_name)
    os.chdir(path)

    threshold_chi_sq = calculate_threshold_chi_sq(calibrated_parameters, calibrated_chi_sq)

    time_list = []
    for parameter_label in settings["parameter_labels_for_ppl"]:
        time = calculate_ppl(parameter_label,
                             calibrated_parameters,
                             calibrated_chi_sq,
                             threshold_chi_sq)
        time_list.append(time)

    print('')
    print('Total time (minutes): ' + str(round(sum(time_list), 4)))
    print('All ppl times (minutes): ' + str(time_list))
    print('Average time per parameter (minutes): ' + str(round(np.mean(time_list), 4)))
    print('SD (minutes): ' + str(round(np.std(time_list), 4)))
        