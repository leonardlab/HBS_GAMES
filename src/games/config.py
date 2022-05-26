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
    #Define settings for this run
    folder_name = 'test'
    modelID = 'synTF'
    dataID = 'synTF dose response'
        
    parameters = [1, 2]
    parameter_labels = ['g', 'a'] 
    free_parameter_labels = ['g'] 
    num_parameter_sets_global_search = 10
    num_parameter_sets_optimization = 2
    
    
    #Define the parameter estimation problem 
    bounds_orders_of_magnitude = 3
    free_parameters = []
    for i, value in enumerate(parameters):
        label = parameter_labels[i]
        if label in free_parameter_labels:
            free_parameters.append(value)
    
    num_free_params = len(free_parameters)
    bounds_log = []
    for i in range(0, num_free_params):
        min_bound = log10(free_parameters[i]) - bounds_orders_of_magnitude
        max_bound = log10(free_parameters[i]) + bounds_orders_of_magnitude
        bounds_log.append([min_bound, max_bound])

    parameter_estimation_problem_definition = {'num_vars': num_free_params,  #set free parameters and bounds
                                               'names': free_parameter_labels, 
                                               'bounds': bounds_log} #bounds are in log scale

class ExperimentalData(Settings):
    if Settings.modelID == 'synTF_chem':
        if Settings.dataID == 'ligand dose response':
            x = np.logspace(0, 3, 5)
            exp_data = [1] * 5
            exp_error = [0.1] * 5
            x_label = 'Ligand (nM)'
            x_scale = 'log'
            y_label = 'Rep. protein (au)'
            
    elif Settings.modelID == 'synTF':
        if Settings.dataID == 'synTF dose response':
            x = np.logspace(0, 2, 5)
            exp_data = [.5] * 5
            exp_error = [0.1] * 5
            x_label = 'synTF (ng)'
            x_scale = 'linear'
            y_label = 'Rep. protein (au)'

class Context(Settings):
    results_folder_path = './Results/'
    date = date.today()
    folder_path = results_folder_path + str(date) + ' ' + Settings.folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
class Save(Settings):
    def createFolder(sub_folder_name):
        ''' 
        Purpose: Create a new folder.
            
        Inputs: 
            sub_folder_name: a string defining the name of the folder to make
           
        Output: 
            path leading to new folder
        '''
        
        path = Context.folder_path + '/' + sub_folder_name
        os.makedirs(path)
            
        return path
    
    def saveConditions():  
        with open('CONDITIONS' + '.txt', 'w') as file:
            file.write('dataID: ' + str(Settings.dataID) + '\n')
            file.write('\n')
            file.write('modelID: ' + str(Settings.modelID) + '\n')
            file.write('\n')
            file.write('parameter_estimation_problem_definition:' + str(Settings.parameter_estimation_problem_definition) + '\n')
            file.write('\n')
   

    
        

        
        