#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:13:10 2020

@author: kate
"""

import os

def makeMainDir(folder_name):    
    ''' 
    Purpose: Create the main directory to hold results of the run
        
    Inputs: 
        folder_name: a string defining the name of the folder
        machine: a string defining the machineID 
                (will be used to set the location of the main Results directory) 
       
           
    Output: 
        results_folder_path + folder_name: a string defining the location of the new folder
    '''

    results_folder_path = '/Users/kate/Documents/GitHub/GAMES/Results/'
    
    try:
        if not os.path.exists(results_folder_path + folder_name):
            os.makedirs(results_folder_path + folder_name)
           
    except OSError:
        print ('Directory already exists')

    print('Main directory created at: ' + folder_name)
    print('***')

    return results_folder_path + folder_name

def createFolder(directory):
    ''' 
    Purpose: Create a new folder.
        
    Inputs: 
        directory: a string defining the name of the folder to make
        machine: a string defining the machineID 
                (will be used to set the location of the main Results directory) 
       
    Output: None
    '''
    
    try:
        if not os.path.exists('./' + directory):
            os.makedirs('./' +directory)
    except OSError:
        print ('Error: Creating directory. ' +  './' + directory)

def saveConditions(conditions_dictionary, initial_params_dictionary, data_dictionary):  
    ''' 
    Purpose: Save conditions, initial params, and data dictionaries
        
    Inputs: 
        directory: a string defining the name of the folder to make
        machine: a string defining the machineID 
                (will be used to set the location of the main Results directory) 
       
    Output: None
    
    Files:
        CONDITIONS.txt (holds the conditions defined in the dictionaries)
    '''

    filename = 'CONDITIONS' 
    with open(filename + '.txt', 'w') as file:
        #f.write('Run ID: ' + str(runID) + '\n')
        file.write('Conditions: ' + str(conditions_dictionary) + '\n')
        file.write('\n')
        file.write('Initial parameters: ' + str(initial_params_dictionary) + '\n')
        file.write('\n')
        file.write('Data: ' + str(data_dictionary) + '\n')
    print('Conditions saved.')

def savePL(threshold_PL_val, calibrated_params, calibrated_chi2, real_param_labels_all):  
    ''' 
    Purpose: Save results of PPL
        
    Inputs: 
       threshold_PL_val: a float defining the threshold chi2 value for PPL calculations
       calibrated_params: a list of floats defining the calibrated parameter set
       calibrated_chi2: a float defining the chi2 value for the calibrated parameter set
       
    Output: None
    
    Files:
        CONDITIONS.txt (holds the conditions defined in the dictionaries)
    '''

    filename = 'CONDITIONS PL' 
    with open(filename + '.txt', 'w') as file:
        #f.write('Run ID: ' + str(runID) + '\n')
        file.write('threshold_PL_val: ' + str(threshold_PL_val) + '\n')
        file.write('\n')
        file.write('parameter_labels: ' + str(real_param_labels_all) + '\n')
        file.write('\n')
        file.write('calibrated_params: ' + str(calibrated_params) + '\n')
        file.write('\n')
        file.write('calibrated_chi2: ' + str(calibrated_chi2) + '\n')
    print('Conditions saved.')
