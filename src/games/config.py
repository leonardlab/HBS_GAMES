#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 08:42:53 2022

@author: kate
"""
import os 
from datetime import date
import numpy as np

class Settings:
    folder_name = 'test'
    modelID = 'synTF'
    dataID = 'synTF dose response'
    free_parameters = [1]
    free_parameter_labels = ['g'] 
    
class ExperimentalData(Settings):
    if Settings.modelID == 'synTF_chem':
        if Settings.dataID == 'ligand dose response':
            x = np.logspace(0, 3, 5)
            exp_data = [1] * 5
            exp_error = [0.1] * 5
            
    elif Settings.modelID == 'synTF':
        if Settings.dataID == 'synTF dose response':
            x = np.logspace(0, 2, 5)
            exp_data = [.5] * 5
            exp_error = [0.1] * 5

class Context(Settings):
    results_folder_path = './Results/'
    date = date.today()
    folder_path = results_folder_path + str(date) + ' ' + Settings.folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
        

        
        