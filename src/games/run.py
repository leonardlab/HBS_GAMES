#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:38:59 2022

@author: kate
"""
import click
from modules.solve_single import Solve_single
from modules.parameter_estimation import estimate_parameters
from modules.parameter_estimation_method_evaluation import evaluate_parameter_estimation_method
from modules.parameter_profile_likelihood import calculate_parameter_profile_likelihood

@click.command()
@click.option('--modules', default='0', help='module number(s) from GAMES workflow as a string - ex: 1 or 12')

def run(modules):
    """Runs the given module(s) from the GAMES workflow with settings defined in the config files

    Parameters
    ----------
    modules
        a string defining the module(s) to run

    Returns
    -------
    None

    """
    
    if '0' in modules:
        print('Starting Module 0...')
        solutions_norm, chi_sq, r_sq = Solve_single.plot_single_parameter_set()
        print('Module 0 completed')
    
    if '1' in modules:
        print('Starting Module 1...')
        evaluate_parameter_estimation_method()
        print('Module 1 completed')
        
    if '2' in modules:
        print('Starting Module 2...')
        calibrated_chi_sq, calibrated_parameters = estimate_parameters()
        print('Module 2 completed')
      
    if '3' in modules:
        print('Starting Module 3...')
        calculate_parameter_profile_likelihood(calibrated_chi_sq, calibrated_parameters)
        print('Module 3 completed')
        

if __name__ == '__main__':
    run()