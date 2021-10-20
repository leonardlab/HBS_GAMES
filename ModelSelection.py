#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 08:25:13 2021

@author: kate
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Settings


#Define filepaths for results of each model for comparison
modelA_filepath = '/Users/kate/Documents/GitHub/GAMES/Results/Results for GAMES manuscript/Model A 1000 + 100/'
modelB_filepath = '/Users/kate/Documents/GitHub/GAMES/Results/Results for GAMES manuscript/Model B 1000 + 100/'
modelC_filepath = '/Users/kate/Documents/GitHub/GAMES/Results/Results for GAMES manuscript/Model C 1000 + 100/'
modelD_filepath = '/Users/kate/Documents/GitHub/GAMES/Results/Results for GAMES manuscript/Model D 1000 + 100/'
filepaths = [modelA_filepath, modelB_filepath, modelC_filepath, modelD_filepath]

#Must set model to B, C, or D in Settings.py for this code to run


def plotModelComparison(filepaths):
    '''
    Purpose: Compare models and visualize results
    
    Inputs: 
        filepaths: a list of strings (length = # models for comparison), eachs string defining the filepath containing the results of the GAMES run
   
    Outputs: None
    
    Figures:
        'MODEL SELECTION PANEL.svg' (plots of training data/simulated data, cost function, AIC across each model for comparison)
        
    '''
    
    def ConvertStringToList(string):
        string = string.replace('[', '')
        string = string.replace(']', '')
        li = list(string.split(", "))
        li = [float(i) for i in li]
        return li
    
    def grabData(filepath):
        filepath = filepath + 'MODULE 2 - FIT TO EXPERIMENTAL DATA/OPT RESULTS.xlsx'
        df = pd.read_excel(filepath)
        
        #Sort by chi2
        df = df.sort_values(by=['chi2'], ascending = True)
        sim = ConvertStringToList(df['Simulation results'].iloc[0])
        L = sim[:11]
        DBD20 = sim[11:19]
        DBD10 = sim[19:]
        chi2 = float(df['chi2'].iloc[0])
        R_sq = float(df['Rsq'].iloc[0])
        
        return L, DBD20, DBD10, R_sq, chi2
    
    def calcChi2(exp, sim, std):
        ''' 
        Purpose: 
            Calculate chi2 between 2 datasets with measurement error described by std
            
        Inputs: 
            exp: experimental data (list of floats, length = # datapoints)
            sim: simulated data (list of floats, length = # datapoints)
            std: meaasurement error for exp data (list of floats, length = # datapoints)
               
        Output: 
            chi2: chi2 value (float) 
        
        '''
    
        #Initialize chi2
        chi2 = 0
        
        #Calculate chi2
        for i, sim_val in enumerate(sim): #for each datapoint
            err = ((exp[i] - sim_val) / (std[i])) ** 2
            chi2 = chi2 + err
            
        return chi2
    
    #Import experimental data and error
    conditions_dictionary, initial_params_dictionary, data_dictionary = Settings.init()
    exp_data = list(data_dictionary["exp_data"])
    error = list(data_dictionary["error"])
    exp_ligand = exp_data[:11]
    exp_20 = exp_data[11:19]
    exp_10 = exp_data[19:]
       
    error_L = error[:11]
    error_20 = error[11:19]
    error_10 = error[19:]
    
    #Grab data from files
    L_ = []
    DBD20_ = []
    DBD10_ = []
    Rsq_ = []
    chi2_ = []
    for filepath in filepaths:
        L, DBD20, DBD10, R_sq, chi2 = grabData(filepath)
        L_.append(L)
        DBD20_.append(DBD20)
        DBD10_.append(DBD10)
        Rsq_.append(R_sq)
        chi2_.append(chi2)
        

    '''Ligand dose response'''
    
    fig = plt.figure(figsize = (8,5))
    ax1 = plt.subplot(231)  
    
    doses_ligand = [0] + list(np.logspace(0, 2, 10))
    ax1.errorbar(doses_ligand, exp_ligand, color = 'black', marker = 'o', yerr = error_L, markerSize = 5, fillstyle = 'none', linestyle = 'none',capsize = 2, label = 'TD')
    
    colors = ['#45b6fe', '#3792cb', '#296d98', '#1c4966']
    labels = ['A', 'B', 'C', 'D']
    
    for i in range(0, len(L_)):
        ax1.plot(doses_ligand, L_[i],  color = colors[i], marker = None, linestyle = ':', label = labels[i])
    
    ax1.set_xlabel('Ligand dose (nM)')
    ax1.set_ylabel('Reporter expression (a.u.)')
    ax1.set_xscale('symlog')
    ax1.legend(frameon = True)
    
    '''DBD characteristic curve'''
    ax2 = plt.subplot(232)
    doses_dbd = [0, 2, 5, 10, 20, 50, 100, 200]                       
    ax2.errorbar(doses_dbd, exp_20, color = 'black', marker = 'o', yerr = error_20, markerSize = 5, fillstyle = 'none', linestyle = 'none',capsize = 2, label = 'Training data')
    
    for i in range(0, len(DBD20_)):
        if i > 0:
            ax2.plot(doses_dbd, DBD20_[i],  color = colors[i], marker = None, linestyle = ':', label = labels[i])
            ax2.plot(doses_dbd, DBD10_[i],  color = colors[i], marker = None, linestyle = '--', label = labels[i])
    ax2.set_xlabel('DBD plasmid (ng)')
    ax2.errorbar(doses_dbd, exp_10, color = 'dimgrey', marker = 'o', yerr = error_10, markerSize = 5, fillstyle = 'none', linestyle = 'none',capsize = 2, label = 'Training data')
    
    
    '''R_sq'''
    #Bar plot with R_sq vals
    ax3 = plt.subplot(233)
    width = 0.5
    ax3.bar(labels, Rsq_, width, color = colors)
    ax3.set_ylabel('R_sq')
    ax3.set_ylim([0.99, 1.0])

    
    '''AIC'''
    #Calculate aic from chi2 - for full dataset (no model A)
    num_free_params = [6,6,5,4]
    aic_vals = []
    for i in range(0, len(chi2_)):
        if i > 0:
            aic = 2 * num_free_params[i] + chi2_[i]
            aic_vals.append(aic)
    min_aic = min(aic_vals)
    delta_aic = [i - min_aic for i in aic_vals]
    labels = ['B', 'C', 'D']
    ax4 = plt.subplot(234)
    ax4.bar(labels, delta_aic, width, color = colors[1:])
    ax4.set_ylabel('delta AIC')
    
    
    #Calculate aic from chi2 - for ligand dose response only
    chi2_L_only = []
    std = [.05] * len(L_[0])
    for i, sim_L in enumerate(L_):
        chi2 = calcChi2(exp_ligand, sim_L, std)
        chi2_L_only.append(chi2)
    
    
    aic_vals = []
    for i in range(0, len(chi2_L_only)):
        aic = 2 * num_free_params[i] + chi2_L_only[i]
        aic_vals.append(aic)
    min_aic = min(aic_vals)   
    delta_aic_L = [i - min_aic for i in aic_vals]
        
    ax5 = plt.subplot(235)
    labels = ['A','B', 'C', 'D']
    ax5.bar(labels, delta_aic_L, width, color = colors)
    ax5.set_ylabel('delta AIC')
    
    #Calculate aic from chi2 - for DBD dose response only
    chi2_DBD_only = []
    std = [.05] * 16
    exp_DBD = exp_20 + exp_10
    for i, list_ in enumerate(DBD20_):
        if i>0:
            sim_DBD = DBD20_[i] + DBD10_[i]
            chi2 = calcChi2(exp_DBD, sim_DBD, std)
            chi2_DBD_only.append(chi2)
    
    aic_vals = []
    num_free_params = num_free_params[1:]
    for i in range(0, len(chi2_DBD_only)):
        aic = 2 * num_free_params[i] + chi2_DBD_only[i]
        aic_vals.append(aic)

    min_aic = min(aic_vals)   
    delta_aic_DBD = [i - min_aic for i in aic_vals]
        
    ax6 = plt.subplot(236)
    labels = ['B', 'C', 'D']
    ax6.bar(labels, delta_aic_DBD, width, color = colors)
    ax6.set_ylabel('delta AIC')
   
    plt.savefig('./Results/MODEL SELECTION PANEL.svg', dpi = 600)

plotModelComparison(filepaths)

    


