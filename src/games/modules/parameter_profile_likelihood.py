#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:24:39 2022

@author: kate
"""
import os
from math import sqrt, log10
from typing import Tuple 
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.set_model import model
from modules.optimization import Optimization
from config import Settings, ExperimentalData
from modules.solve_single import Solve_single
from analysis.metrics import calc_chi_sq
from modules.global_search import generate_parameter_sets, solve_global_search
from utilities.saving import create_folder, save_conditions

def calculate_parameter_profile_likelihood(
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
        path = create_folder(sub_folder_name)
        os.chdir(path)
        save_conditions()

        threshold_chi_sq = calculate_threshold_chi_sq(calibrated_parameters, calibrated_chi_sq)
        time_list = []
        for parameter_label in Settings.parameter_labels:
            df, time = calculate_profile_likelihood(parameter_label, 
                                              calibrated_parameters, 
                                              calibrated_chi_sq, 
                                              threshold_chi_sq)
            
            time_list.append(time)
            
        print(time_list)
        print('Total time (minutes): ' + str(round(sum(time_list), 4)))
        print('Average time per parameter (minutes): ' + str(round(np.mean(time_list), 4)))
        print('SD (minutes): ' + str(round(np.std(time_list), 4)))


def calculate_threshold_chi_sq(calibrated_parameters: list, calibrated_chi_sq: float) -> float:
    """Calculates threshold chi_sq value for PPL calculations

    Parameters
    ----------
    calibrated_parameters
        a list of floats containing the calibrated values for each parameter

    calibrated_chi_sq
        a float defining the chi_sq associated with the calibrated parameter set

    Returns
    -------
    threshold_chi_sq
        a float defining the threshold chi_sq value

    """
    
    def calculate_chi_sq_fit() -> list:
        """Runs optimization on each noise realization with the calibrated parameters as the initial guess
    
        Parameters
        ----------
        None
    
        Returns
        -------
        chi_sq_fit_list
            a list of floats defining the chi_sq_fit values for each noise realization
    
        """
        
        df_noise = pd.DataFrame()
        for i, parameter_label in enumerate(Settings.parameter_labels):
            df_noise[parameter_label] = [calibrated_parameters[i]] * n_noise_realizations
        df_noise["data"] = exp_data_noise_list
        _, _, df_optimization_results, _ = Optimization.optimize_all(df_noise, "PPL")
        chi_sq_fit_list = list(df_optimization_results["chi_sq"])
        
        return chi_sq_fit_list

    def plot_chi_sq_distribution() -> None:
        """Plots threshold chi_sq value for PPL calculations
    
        Parameters
        ----------
        None
        
        Returns
        -------
        None
    
        """
    
        plt.figure(figsize=(5, 3))
        plt.xlabel("chi_sq_ref - chi_sq_fit")
        plt.ylabel("Count")
        y, x, _ = plt.hist(chi_sq_distribution, bins=35, histtype="bar", color="dimgrey")
        plt.plot(
            [threshold_chi_sq, threshold_chi_sq],
            [0, max(y)],
            "-",
            lw=3,
            alpha=0.6,
            color="dodgerblue",
            linestyle=":",
            label=str(confidence_interval_label) + "%",
        )
        plt.savefig("./CHI_SQ DISTRIBUTION", bbox_inches="tight", dpi=600)
    
        print("******************")
        threshold_chi_sq_rounded = np.round(threshold_chi_sq, 1)
        print(
            "chi_sq THRESHOLD FOR "
            + str(len(Settings.parameters))
            + " PARAMETERS WITH "
            + str(confidence_interval_label)
            + "% CONFIDENCE"
        )
        print(threshold_chi_sq_rounded)
        print("******************")
        
    def save_chi_sq_distribution() -> None: 
        """Saves threshold chi_sq value for PPL calculations
    
        Parameters
        ----------
        None
        
        Returns
        -------
        None
    
        """
        filename = "CONDITIONS PL"
        with open(filename + ".txt", "w") as file:
            file.write("threshold_chi_sq: " + str(threshold_chi_sq) + "\n")
            file.write("\n")
            file.write("parameter_labels: " + str(Settings.parameter_labels) + "\n")
            file.write("\n")
            file.write("calibrated_params: " + str(calibrated_parameters) + "\n")
            file.write("\n")
            file.write("calibrated_chi_sq: " + str(calibrated_chi_sq) + "\n")
        print("Conditions saved.")

    
    p_ref = [15, 1, 0.05, 720, 100, 2]
    print("REFERENCE PARAMETERS - ORIGINAL EXPERIMENTAL DATA")
    model.parameters = p_ref
    norm_solutions_ref, chi_sq_ref, _ = Solve_single.solve_single_parameter_set()
    print("chi_sq REFERENCE: " + str(round(chi_sq_ref, 4)))
    print("******************")
    
    # Generate noise realizations and calculate chi_sq_ref for each noise realization
    n_noise_realizations = 10
    exp_data_original = ExperimentalData.exp_data
    exp_data_noise_list = []
    chi_sq_ref_list = []

    # Define mean and standard error for error distribution
    mu = 0
    sigma = 0.05 / sqrt(3)

    # Generate noise array
    np.random.seed(6754)
    noise_array = np.random.normal(mu, sigma, (n_noise_realizations, len(exp_data_original)))

    for i in range(0, n_noise_realizations):
        # add noise and append to experimental data to list for saving
        exp_data_noise = []
        for j, val in enumerate(exp_data_original):
            new_val = val + noise_array[i, j]
            new_val = max(new_val, 0.0)
            exp_data_noise.append(new_val)
            
        #re-normalize data
        exp_data_noise_norm = [i / max(exp_data_noise) for i in exp_data_noise]
        exp_data_noise_list.append(exp_data_noise_norm)

        # Calculate chi_sq_ref using noise realization i
        chi_sq = calc_chi_sq(norm_solutions_ref, exp_data_noise_norm, ExperimentalData.exp_error)
        chi_sq_ref_list.append(chi_sq)

    print("Calculating chi_sq FIT...")
    chi_sq_fit_list = calculate_chi_sq_fit()
    
    # Plot chi_sq distribution and calculate 99% CI threshold
    chi_sq_distribution = []
    for i, ref_val in enumerate(chi_sq_ref_list):
        chi_sq_distribution.append(ref_val - chi_sq_fit_list[i])
    confidence_interval_label = Settings.confidence_interval * 100
    threshold_chi_sq = np.percentile(chi_sq_distribution, confidence_interval_label)
    plot_chi_sq_distribution() 
    save_chi_sq_distribution() 

    return threshold_chi_sq


def calculate_profile_likelihood(
    parameter_label: str, calibrated_parameter_values: list, calibrated_chi_sq: float, threshold_chi_sq: float
) -> Tuple[pd.DataFrame, float]:
    """Calculates the PPL for the given parameter

    Parameters
    ----------
    parameter_label
        a string defining the parameter label

    calibrated_parameter_values
        a list of floats containing the calibrated values for each parameter

    calibrated_chi_sq
        a float defining the chi_sq associated with the calibrated parameter set
        
    threshold_chi_sq
        a float defining the threshold chi_sq value

    Returns
    -------
    df
        a df containing the PPL results for the given parameter
        
    elapsed_time_total
        a float defining the time to calculate the PPL for the given parameter (in minutes)

    """
    def calculate_chi_2_PPL_single_datapoint(param_vals: list, fixed_index: int, fixed_val: float, fixed_index_in_free_parameter_list: int) -> Tuple[float, float, list]:
        '''
        Calculates chi_sq_PPL for a single datapoint on the profile likelihood
        
        Parameters
        ---------- 
        param_vals
            a list of parameter values (initial guesses)
            
        fixed_index
            an integer defining the index in param_vals that should be fixed for 
            this calculation, this is the index of the parameter in the list of 
            Settings.parameters
            
        fixed_val
            a float defining the value that the fixed index should be fixed at
            
        fixed_index_in_free_parameter_list
            an integer defining the index in the list of FREE parameters 
            (Settings.free_parameters) that should be fixed for this calculation
            
        Returns
        -------
        fixed_val
            a float defining the value that the fixed index should be fixed at 
            (x-axis value of the datapoint for PPL plots)
            
        calibrated_chi_sq
            a float defining the minimum chi2 attainable for the parameter 
            estimation problem (x-axis value of the datapoint for PPL plots)
            
        calibrated_parameters
            a list of floats defining the parameter set for which the min_chi_sq 
            was obtained
 
        '''
    
        #Define the parameter estimation problem
        problem_PPL = {'num_vars': 0,  
                    'names': [], 
                    'bounds': []} 
    
        #Fix given parameter at fixed index and let other parameters vary
        problem_PPL['num_vars'] = len(Settings.free_parameters) - 1
        problem_PPL['names'] = [x for i,x in enumerate(Settings.parameter_estimation_problem_definition['names']) if i != fixed_index_in_free_parameter_list]
        problem_PPL['bounds'] = [x for i,x in enumerate(Settings.parameter_estimation_problem_definition['bounds']) if i != fixed_index_in_free_parameter_list]
        
        #Run PEM
        df_parameters = generate_parameter_sets(problem_PPL)
        df_global_search_results = solve_global_search(df_parameters)
        _, calibrated_chi_sq, _, calibrated_parameters = Optimization.optimize_all(df_global_search_results)

        return fixed_val, calibrated_chi_sq, calibrated_parameters
    
    def plot_parameter_profile_likelihood():
        '''
        Plots PPL results for a single parameter
        
        Parameters
        ---------- 
        None
            
        Returns
        -------
        None
 
        '''
        #Restructure ucture data
        calibrated_parameter_values_log = [log10(i) for i in calibrated_parameter_values]
        x = fixed_parameter_values_both_directions
        y = chi_sq_PPL_list_both_directions
        
        #Drop data points outside of bounds
        x_plot = []
        y_plot = []
        for j in range(0, len(x)):
            if abs(x[j]) > (10 ** -10):
                x_plot.append(x[j])
                y_plot.append(y[j])
            else:
                print('data point dropped - outside parameter bounds')
                print(x[j])
        x_log = [log10(i) for i in x_plot]
        
        #Plot PPL data
        fig = plt.figure(figsize = (3,3))
        plt.plot(x_log, y_plot, 'o-', color = 'black', markerSize = 4, fillstyle='none', zorder = 1)
        plt.xlabel(parameter_label)
        plt.ylabel('chi_sq')
        
        #Plot calibrated parameter value as a blue dot
        cal_vals = []
        for i_ in range(0, len(Settings.parameter_labels)): #for each parameter in p_all
            for j in range(0, len(Settings.free_parameter_labels)): #for each free param
                if Settings.parameter_labels[i_] == Settings.free_parameter_labels[j]: #if parameter is a free parameter 
                    cal_vals.append(calibrated_parameter_values_log[i_]) #replace with calibrated val
        x = [cal_vals[fixed_index]]
        y = [calibrated_chi2]
        ax.scatter(x, y, s=16, marker='o', color = 'dodgerblue', zorder = 2)
        plt.ylim([calibrated_chi2 * .75, threshold_val * 1.2])
         
        #Plot threshold as dotted line
        x1 = [min(x_log), max(x_log)]
        y1 = [threshold_val, threshold_val]
        ax.plot(x1, y1, ':', color = 'dimgrey')
            
        plt.savefig('PROFILE LIKELIHOOD PLOT ' + parameter_label + '.svg')

    start_time = datetime.datetime.now() #Record the start time
    
    #Determine index for PPL based on parameter_label
    #These are the indicies of the free parameters in the list Settings.parameter_labels
    for i, label in enumerate(Settings.parameter_labels):
        if label == parameter_label:
            fixed_index = i
            
    #Determine the index in Settings.free_parameter_labels that corresponds to the index in Settings.parameter_labels
    for j, label in enumerate(Settings.free_parameter_labels):
        if label == parameter_label:
            fixed_index_in_free_parameter_list = j #The index of the fixed parameter (for this PL run) in the list of free parameters
 
    #Set the minimum and maximum acceptable step choices (chi2_PL(current step) - chi2_PL(last step)) 
    #set the min step to 1% of starting value
    min_step_list = [.01 * i for i in calibrated_parameter_values] 
    
    #set the max step to 20% of starting value
    max_step_list = [.2 * i for i in calibrated_parameter_values] 
    
    #Set the target value for PL difference between steps (chi2_PL(current step) - chi2_PL(last step)) 
    #and acceptable range
    q = .1 
    target_val = threshold_chi_sq * q
    min_threshold_limit = target_val * .1
    max_threshold_limit = target_val * 2
    max_steps = 50  #Set maximum number of steps for EACH direction
    
    #Set min and max flags to default values 
    min_flag = False
    max_flag = False
    
    num_evals = 0 #number of evaluations (full PEM runs)
    
    #Define the min and max bounds of the fixed parameter, convert to linear scale
    min_bound_log = Settings.parameter_estimation_problem_definition['bounds'][fixed_index_in_free_parameter_list][0]
    max_bound_log = Settings.problem_free['bounds'][fixed_index_in_free_parameter_list][1]
    min_bound = 10 ** min_bound_log
    max_bound = 10 ** max_bound_log
    
    print('******************')
    print('Starting profile likelihood calculations for ' + 
          parameter_label + '...')
    
    for direction in [-1, 1]: 
        print('')
        if direction == -1:
            print('negative direction')
        elif direction == 1:
            print('positive direction')
            
        #Set default min and max step vals for the fixed parameter
        min_step = min_step_list[fixed_index] 
        max_step = max_step_list[fixed_index] 
        
        #Set parameter-specific PPL hyperparameters if relevant
        if parameter_label == "k_bind" and direction == -1:
             min_step = .01 * min_step 
        if parameter_label == "m" and direction == -1:
             min_step = .01 * min_step #m
        if parameter_label == "m" and direction == 1:
             max_step = 3 * max_step #m
             max_steps = 100
        
        print('min step: ' + str(min_step))
        print('max step: ' + str(max_step))

        #Initialize lists to hold results for this PPL only
        chi_sq_PPL_list  = [] #chi_sq_PPL values
        fixed_parameter_values = []  #fixed parameter values, linear scale
        all_parameter_values = [] #lists of all parameter values
       
        #Initialize counters
        step_number = 0
        attempt_number = 0

        while step_number < max_steps:
            print('******************')
            print('step_number: ' + str(step_number))
            print('attempt_number: ' + str(attempt_number))
            print('')
    
            #If this is the first step
            if step_number == 0: 
                
                #Set the fixed value as the calibrated value
                fixed_val = calibrated_parameter_values[fixed_index]
                print('starting val: ' + str(fixed_val))
                
                #Set the min_ for the binary step to 0 and the max_ to the max_step
                min_ = 0
                max_ = max_step
                
                print('min_: ' + str(min_))
                print('max_: ' + str(max_))
                
                #Set the min and max flags to false
                min_flag = False
                max_flag = False
           
            #If this is not the first step
            else: 
                
                #If the step value is more than or equal to the max_step, 
                #then replace the step with #the max step and set max_flag to True
                if step_val >= max_step: 
                    step_val = max_step
                    max_flag = True
                    print('step replaced with max step')
                   
                #If the step value is less than or equal to the min_step, 
                #then replace the step with the min step and set min_flag to True
                elif step_val <= min_step: 
                    step_val = min_step
                    min_flag = True
                    print('step replaced with min step')
                  
                #If the step is between the min and max step limits, set flags to 
                #False and continue
                else:
                    min_flag = False
                    max_flag = False
                 
                #Determine the next fixed value
                fixed_val = fixed_parameter_values[step_number - 1] + (direction * step_val)
                print('previous val: ' + str(round(fixed_parameter_values[step_number - 1], 8))) 
                print('step val: ' + str(round(step_val, 8)))
                print('fixed val: ' + str(round(fixed_val)))
                
                if fixed_val <= min_bound: #if fixed val is negative or less than the min bound of the parameter
                    print('fixed val is negative')
                    
                    #if this is the minimum step, break (cannot take a smaller step 
                    #to reach a > 0 fixed value)    
                    if min_flag == True: 
                        print('negative or zero fixed value reached - break')
                        break #out of top level while loop and start next direction/parameter
                    
                    #if this is not the minimum step, try with half the last step
                    else: 
                        while fixed_val <= min_bound:
                            print('try again')
                            #reset the max step to be half what it was before
                            max_step = max_step/2 
                            
                            #try a step val half the size as the previous step
                            step_val = step_val/2 
                            
                            #determine fixed value
                            fixed_val = fixed_parameter_values[step_number - 1] + (direction * step_val) 
                            print('fixed val: ' + str(fixed_val))
                            
                            if step_val <= min_step:
                                step_val = min_step
                                fixed_val = fixed_parameter_values[step_number - 1] + (direction * step_val) 
                                min_flag = True
                                print('step replaced with min step')
                                break #break out of this while loop
                            
                if fixed_val <= 0.0:
                    print('negative or zero fixed value reached - break')
                    break #out of top level while loop and start next direction/parameter
          
                print('new val: ' + str(round(fixed_val, 4))) #linear
            
         
            #Restructure parameter set to feed into calculate_chi_2_PPL_single_datapoint
            params_for_opt = Settings.parameters
            for i in range(0, len(Settings.parameter_labels)): #for each parameter in p_all
                for j in range(0, len(Settings.free_parameter_labels)): #for each free parameter
                
                    #if parameter is a free parameter, replace with calibrated 
                    if Settings.parameter_labels[i] == Settings.free_parameter_labels[j]: 
                        if i == fixed_index:
                            params_for_opt[i] = fixed_val #replace with fixed val
                        else:   
                            params_for_opt[i] = calibrated_parameter_values[i] #Replace with cal val
           
            param_val, chi_sq_PPL_val, param_vals = calculate_chi_2_PPL_single_datapoint(params_for_opt, 
                                                               fixed_index, fixed_val, fixed_index_in_free_parameter_list)
            print('chi2_PL: ' + str(round(chi_sq_PPL_val, 3)))
            num_evals += 1
            print('calibrated parameters: ' + str(param_vals))
            
            #Determine whether to accept step or try again with new step size

            #if this is not the first step (calibrated parameter value)
            if step_number != 0:
                
                #Calculate difference in PL between current value and previous value
                PPL_difference = abs(chi_sq_PPL_val - chi_sq_PPL_list [step_number -1])
                print('PPL_differenceerence: ' + str(round(PPL_difference, 3)))
                print('')
               
                #if PPL_difference is in between the min and max limits for PL difference
                if PPL_difference <= max_threshold_limit and PPL_difference >= min_threshold_limit: 
                    
                    #if the PL value is greater than or equal to the threshold value, but less 
                    #than 1.1 * the threshold value
                    if chi_sq_PPL_val >= threshold_chi_sq and chi_sq_PPL_val <= 1.1 * threshold_chi_sq: 
                        
                        #Record values and break loop 
                        chi_sq_PPL_list .append(chi_sq_PPL_val)
                        fixed_parameter_values.append(param_val)
                        all_parameter_values.append(param_vals)
                        print('break - Threshold chi2 reached')
                        break
                    
                    #otherwise, if the PL value is greater than 1.1 * the threshold value, 
                    #then the step is too large
                    elif chi_sq_PPL_val > 1.1 * threshold_chi_sq:
                        
                        #If this is the minimum step, then a smaller step cannot be taken
                        if min_flag == True:
                            
                            #Record values and break loop 
                            chi_sq_PPL_list .append(chi_sq_PPL_val)
                            fixed_parameter_values.append(param_val)
                            all_parameter_values.append(param_vals)
                            print('break - Threshold chi2 reached')
                            break
                          
                        #If this is not the minimum step, then a smaller step should be used
                        else:
                            #Set the max bound for the binary step to be equal to 
                            #the current step
                            max_ = step_val
                            
                            #Calculate the next step value (binary step algorithm)
                            step_val  = (min_ + max_)/2
                            
                            #increase the attempt number
                            attempt_number += 1
                            
                            print('step rejected - too large')  
                            print('')
                            print('new bounds')
                            print('min_: ' + str(min_))
                            print('max_: ' + str(max_))
                            print('new step val: ' + str(step_val))
                    
                    #Otherwise, if the fixed parameter hits the min or max bound
                    elif param_val > max_bound or param_val < min_bound: 
                        
                        #Record values and break loop 
                        chi_sq_PPL_list .append(chi_sq_PPL_val)
                        fixed_parameter_values.append(param_val)
                        all_parameter_values.append(param_vals)
                        print('break - Parameter bound reached')
                        break
                    
                    #Otherwise, accept the step and record the results 
                    #(the parameter is not above the threshold and does not 
                    #reach the parameter bound)
                    else: 
                        print('step accepted')
                        
                        #Record results
                        chi_sq_PPL_list .append(chi_sq_PPL_val)
                        fixed_parameter_values.append(param_val)
                        all_parameter_values.append(param_vals)
            
                        #increase the step number
                        step_number += 1
                        
                        #Reset the attempt counter
                        attempt_number = 0
                        
                        #Set min_ bound back to 0 (no step from the previously recorded value)
                        min_ = 0
                
                #if the step size is the maximum step and the PL difference is still too low
                elif max_flag == True and PPL_difference < min_threshold_limit: 
                   
                    #If the parameter value is above the max bound or below the min bound
                    if fixed_val > max_bound or fixed_val < min_bound:
                    
                        #Record results and break loop
                        chi_sq_PPL_list .append(chi_sq_PPL_val)
                        fixed_parameter_values.append(param_val)
                        all_parameter_values.append(param_vals)
                        print('break - Parameter bound reached')
                        break
                    
                    #if the PL value is more than or equal to the threshold value and is less 
                    #than 1.1 * the threshold value
                    elif chi_sq_PPL_val >= threshold_chi_sq and chi_sq_PPL_val <= 1.1 * threshold_chi_sq: 
                        
                        #Record results and break loop
                        chi_sq_PPL_list .append(chi_sq_PPL_val)
                        fixed_parameter_values.append(param_val)
                        all_parameter_values.append(param_vals)
                        print('break - Threshold chi2 reached')
                        break
                    
                    #otherwise, if the step size is the maximum step AND the PL difference 
                    #is still too low AND the threshold value is not met AND and the 
                    #parameter bounds are not reached, then accept the step by default
                    else: 
                        #Record results
                        chi_sq_PPL_list .append(chi_sq_PPL_val)
                        fixed_parameter_values.append(param_val)
                        all_parameter_values.append(param_vals)
                        
                        #Increasethe step counter
                        step_number += 1
                        
                        #Reset the attempt counter
                        attempt_number = 0
        
                        print('step accepted by default - max step value reached')
                        
                        #Set min_ bound for step size calculations equal 0 (no step)
                        min_ = 0
                        
                        #Keep the step value at the maximum step value
                        step_val = max_step
                    
                #if the parameter reaches the min step, but the PL difference is still too high
                elif min_flag == True and PPL_difference > max_threshold_limit: 
                    
                    #if the PL value is more than or equal to the threshold value and is 
                    #less than 1.1 * the threshold value
                    if chi_sq_PPL_val >= threshold_chi_sq:  
                        
                        #Record results and break loop
                        chi_sq_PPL_list .append(chi_sq_PPL_val)
                        fixed_parameter_values.append(param_val)
                        all_parameter_values.append(param_vals)
                        print('break - Threshold chi2 reached')
                        print('min_step used - cannot take a smaller step')
                        break
        
                    #if parameter hits bound
                    elif fixed_val > max_bound or fixed_val < min_bound: 
                        
                        #Record results and break loop
                        chi_sq_PPL_list .append(chi_sq_PPL_val)
                        fixed_parameter_values.append(param_val)
                        all_parameter_values.append(param_vals)
                        print('break - Parameter bound reached')
                        print('min_step used - cannot take a smaller step')
                        break
             
                    #if parameter is not above the threshold or the parameter bound, 
                    #accept the step and record the results by default
                    else: 
                        #Record results
                        chi_sq_PPL_list .append(chi_sq_PPL_val)
                        fixed_parameter_values.append(param_val)
                        all_parameter_values.append(param_vals)
                        
                        #Add 1 to the step counter
                        step_number += 1
                        
                        #Reset the attempt counter
                        attempt_number = 0
                        print('step accepted by default - min step value reached')
                        
                        #Set min_ bound for step size calculations equal to the current step val
                        min_ = 0
                        step_val = min_step
               
                #if conditions are not met because PL difference is too large, 
                #try again with a smaller step size (or if a differenence is negative, 
                #then step is too large and need to try a smaller one if possible)
                elif PPL_difference > max_threshold_limit or PPL_difference < 0: 
    
                    #Set the max bound for the next step to be equal to the current step
                    max_ = step_val
                    
                    #Calculate the next step value
                    step_val  = (min_ + max_)/2
                    
                    #If min and max for binary step calculations are equal, then set the step 
                    #val to default to the min step
                    if min_ == max_: 
                        step_val = min_step
                        print('step replaced with min step')
                    
                    #add 1 to the attempt number
                    attempt_number += 1
                    
                    print('step rejected - too large')  
                    print('')
                    print('new bounds')
                    print('min_: ' + str(min_))
                    print('max_: ' + str(max_))
                    print('new step val: ' + str(step_val))
                   
                #if conditions are not met because PL difference is too small, try again with a 
                #larger step size
                elif PPL_difference < min_threshold_limit:
                    #Set the min bound for the next step to be equal to the current step
                    min_ = step_val 
                    
                    #Calculate the next step value
                    step_val  = (min_ + max_)/2
                    
                    #If min and max for binary step calculations are equal, then set the 
                    #step val to default to half of the max step
                    if min_ == max_:
                        step_val = max_step / 2
                        print('step replaced with max step')
                    
                    #Increase the attempt number
                    attempt_number += 1
                    
                    print('step rejected - too small')  
                    print('')
                    print('new bounds')
                    print('min_: ' + str(min_))
                    print('max_: ' + str(max_))
                    print('new step val: ' + str(step_val))

                else:
                    print('else')
                      
            #if is the calibrated param (0th step), record results by default and start with the 
            #max step value
            elif step_number == 0: 
       
                #Set next step to max step value
                step_val  = max_step
                
                #Record results
                if direction == -1:
                    chi_sq_PPL_list .append(chi_sq_PPL_val)
                    fixed_parameter_values.append(param_val)  
                    all_parameter_values.append(param_vals)
                    
                    calibrated_chi_sq = chi_sq_PPL_val
                    param_vals_calibrated = param_vals
                    
                    step_number += 1
                    attempt_number = 0
                    
                elif direction == 1:
                    chi_sq_PPL_list.append(calibrated_chi_sq)
                    print('step accepted - calibrated parameter (Step 0)')
                    print(calibrated_chi_sq)
                    fixed_parameter_values.append(param_val)     
                    all_parameter_values.append(param_vals_calibrated)
                    
                    step_number += 1
                    attempt_number = 0

        #Prepare lists for plotting (reverse the negative direction (left), then append the 
        #positive direction (right))
        if direction == -1: 
            chi_sq_PPL_list .reverse()
            PPL_left = chi_sq_PPL_list 
            fixed_parameter_values.reverse()
            params_left = fixed_parameter_values
            all_parameter_values.reverse()
            param_vals_left = all_parameter_values
            
        elif direction == 1:
            PPL_right = chi_sq_PPL_list 
            params_right = fixed_parameter_values
            param_vals_right = all_parameter_values
            
    #combine LHS and RHS of the PPL
    chi_sq_PPL_list_both_directions = PPL_left + PPL_right 
    fixed_parameter_values_both_directions = params_left + params_right
    all_parameter_values_both_directions = param_vals_left + param_vals_right
    
    print('fixed parameter values:')
    print(fixed_parameter_values_both_directions)
    print('profile likelood values:')
    print(chi_sq_PPL_list_both_directions)
    
    print('***')
    print('number of PEM rounds to evaluate PPL for this parameter: ' + str(num_evals))

    #Record stop time
    stop_time = datetime.datetime.now()
    elapsed_time = stop_time - start_time
    elapsed_time_total = round(elapsed_time.total_seconds(), 1)
    print('Time for this parameter (minutes): ' + str(round(elapsed_time_total/60, 4)))
    print('***')
   
    #Plot results
    plot_parameter_profile_likelihood()
    
    #Structure and save results
    df = pd.DataFrame()
    df['fixed ' + parameter_label] = fixed_parameter_values_both_directions
    df['fixed ' + parameter_label + ' PPL'] = chi_sq_PPL_list_both_directions
    df['fixed ' + parameter_label + ' all parameters'] = all_parameter_values_both_directions

    filename = './PROFILE LIKELIHOOD RESULTS ' + parameter_label + '.xlsx'
    with pd.ExcelWriter(filename) as writer:  # doctest: +SKIP
        df.to_excel(writer, sheet_name = parameter_label)

    return df, elapsed_time_total
    




def plot_parameter_profile_likelihood_consequences(df_profile_likelihood_results):
    pass
