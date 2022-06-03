#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 08:42:53 2022

@author: kate
"""
import os 
from datetime import date
import numpy as np
from math import log10

class Settings:
    """
    Define settings.
    
    """
    #Define settings for this run
    folder_name = 'synTF_chem test'
    modelID = 'synTF_chem'
    dataID = 'ligand dose response'
        
    parameters = [15, .01, 1, 500, 2, 2]
    parameter_labels = ['e', 'b', 'k_bind', 'm', 'km', 'n']
    free_parameter_labels = ['e', 'b', 'k_bind', 'm', 'km', 'n']
    num_parameter_sets_global_search = 10
    num_parameter_sets_optimization = 2
    bounds_orders_of_magnitude = 3
    
    weight_by_error = 'no'
    
    #Define the parameter estimation problem 
    free_parameters = []
    free_parameter_indicies = []
    for i, value in enumerate(parameters):
        label = parameter_labels[i]
        if label in free_parameter_labels:
            free_parameters.append(value)
            free_parameter_indicies.append(i)
    
    num_free_params = len(free_parameters)
    bounds_log = []
    for i in range(0, num_free_params):
        min_bound = log10(free_parameters[i]) - bounds_orders_of_magnitude
        max_bound = log10(free_parameters[i]) + bounds_orders_of_magnitude
        bounds_log.append([min_bound, max_bound])
        
    if free_parameter_labels[-1] == 'n':
        bounds_log[-1] = [0, .602] #n

    parameter_estimation_problem_definition = {'num_vars': num_free_params,  #set free parameters and bounds
                                               'names': free_parameter_labels, 
                                               'bounds': bounds_log} #bounds are in log scale
class ExperimentalData(Settings):
    """
    Define experimental data.
    
    """
    if Settings.modelID == 'synTF_chem':
        if Settings.dataID == 'ligand dose response':
            x = np.logspace(0, 3, 6)
            exp_data_raw = [0, .1, 2, 3.5, 4.7, 5]
            exp_error_raw  = [0.1] * 6
            
            max_val = max(exp_data_raw)
            exp_data = []
            exp_error = []
            for i, val in enumerate(exp_data_raw):
                exp_data.append(val/max_val)
                exp_error.append(exp_error_raw[i]/max_val)
         
            x_label = 'Ligand (nM)'
            x_scale = 'log'
            y_label = 'Rep. protein (au)'
            
    elif Settings.modelID == 'synTF':
        if Settings.dataID == 'synTF dose response':
            x = np.logspace(0, 2, 5)
            exp_data_raw = [.5] * 5
            exp_error_raw = [0.1] * 5
            
            max_val = max(exp_data_raw)
            exp_data = []
            exp_error = []
            for i, val in enumerate(exp_data_raw):
                exp_data.append(val/max_val)
                exp_error.append(exp_error_raw[i]/max_val)
            
            x_label = 'synTF (ng)'
            x_scale = 'linear'
            y_label = 'Rep. protein (au)'
       

class Context(Settings):
    """
    Define context
    
    """
    results_folder_path = './Results/'
    date = date.today()
    folder_path = results_folder_path + str(date) + ' ' + Settings.folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
class Save(Settings):
    """
    Methods for creating folders and saving run conditions
    
    """
    
    def createFolder(sub_folder_name):
        """Create a new folder.
        
        Parameters
        ----------
        sub_folder_name
            string defining the name of the folder to make
            
        Returns
        -------
        path
            path leading to new folder
        """
        
        path = Context.folder_path + '/' + sub_folder_name
        os.makedirs(path)
            
        return path
    
    def saveConditions():  
        """Save conditions for the given run
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        
        """
        
        with open('CONDITIONS' + '.txt', 'w') as file:
            file.write('dataID: ' + str(Settings.dataID) + '\n')
            file.write('\n')
            file.write('modelID: ' + str(Settings.modelID) + '\n')
            file.write('\n')
            file.write('parameter_estimation_problem_definition:' + str(Settings.parameter_estimation_problem_definition) + '\n')
            file.write('\n')
   

    
        

        
        