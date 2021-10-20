#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kateldray
"""
#Package imports
import numpy as np

def defineExp(data, df_ref):
    ''' 
    Purpose: Import the experimental data from an (.xlsx sheet) and 
        structure the data to be compatible with downstream LMFIT code.
    
    Inputs:
        data: a string defining the data identity
            (= 'ligand dose response only' or 'ligand dose response and DBD dose response')
        df_ref: a dataframe containing the reference data
      
        
    Outputs:
         x: a list of lists defining the component doses for each datapoint. 
             Each list has 3 items - [DBD dose, AD dose, ligand dose] - and there 
             are the same number of lists as number of datapoints.
         data: a list of floats defining the normalized reporter expression values 
             for each datapoint (length = # datapoints)
         error: list of floats defining the normalized reporter expression error 
             values for each datapoint (length = # datapoints)
        
  '''

    doses_ligand = [0] + list(np.logspace(0, 2, 10))
    doses_dbd = [0, 2, 5, 10, 20, 50, 100, 200]
    
    #Restructure data
    if data == 'ligand dose response only':
        data_ligand = df_ref['L']
 
        x = []
        for val in doses_ligand:
            x1 = 50
            x2 = 50
            x3 = val
            x.append([x1, x2, x3])
  
        data = data_ligand
        error = [.05] * len(x)
        
    elif data == 'ligand dose response and DBD dose response':
        data_ligand = list(df_ref['L'])
        data_dbd_20 = list(df_ref['DBD_20'])[0:8]
        data_dbd_10 = list(df_ref['DBD_10'])[0:8]
        
        x = []
        for val in doses_ligand:
            x1 = 50
            x2 = 50
            x3 = val
            x.append([x1, x2, x3])
        
        for val in doses_dbd:
            x1 = val
            x2 = 20
            x3 = 100
            x.append([x1, x2, x3])
            
        for val in doses_dbd:
            x1 = val
            x2 = 10
            x3 = 100
            x.append([x1, x2, x3])
        
        data = data_ligand + data_dbd_20 + data_dbd_10
        error = [.05] * len(x)
       
    return x, data, error
