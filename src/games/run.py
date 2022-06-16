#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:38:59 2022

@author: kate
"""
import click
from modules.solve_single import run_single_parameter_set
from modules.parameter_estimation.run_parameter_estimation import run_parameter_estimation
from modules.parameter_estimation_method_evaluation.run_parameter_estimation_method_evaluation import run_parameter_estimation_method_evaluation
from modules.parameter_profile_likelihood.run_parameter_profile_likelihood import run_parameter_profile_likelihood

@click.command()
@click.option('--modules', default='0', help='module number(s) from GAMES workflow as a string - ex: 1 or 12 or 123')

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
        solutions_norm, chi_sq, r_sq = run_single_parameter_set()
        print('Module 0 completed')
        print('')
    
    if '1' in modules:
        print('Starting Module 1...')
        run_parameter_estimation_method_evaluation()
        print('Module 1 completed')
        print('')
        
    if '2' in modules:
        print('Starting Module 2...')
        calibrated_chi_sq, calibrated_parameters = run_parameter_estimation()
        print('Module 2 completed')
        print('')
      
    if '3' in modules:
        print('Starting Module 3...')
        run_parameter_profile_likelihood(calibrated_chi_sq, calibrated_parameters)
        print('Module 3 completed')
        print('')
        

if __name__ == '__main__':
    run()