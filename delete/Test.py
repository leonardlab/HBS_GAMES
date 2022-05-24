#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:34:00 2020

@author: kate
"""
#Package imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#GAMES imports
import Settings
from Solvers import calcChi2, calcRsq, solveSingle
from Run import addNoise, solveAll
from Saving import createFolder

#Define settings
conditions_dictionary, initial_params_dictionary, data_dictionary = Settings.init()
run = conditions_dictionary["run"]
full_path = conditions_dictionary["directory"]
model = conditions_dictionary["model"]
data = conditions_dictionary["data"]
error = data_dictionary["error"]
exp_data = data_dictionary["exp_data"]
plt.style.use('./paper.mplstyle.py')

doses_ligand = [0] + list(np.logspace(0, 2, 10))
doses_dbd = [0, 2, 5, 10, 20, 50, 100, 200]

def saveRefData(data_):
    
    ''' 
    Purpose: Save reference training data
        
    Inputs: 
        data: list of floats, each float describes reporter expression for a different condition 
            (length = # datapoints )
           
    Output:
        df_ref: dataframe containing the reference training data in a structure compatible with 
            downstream GAMES code 
        
    Files: 
        REFERENCE TRAINING DATA.xlsx (dataframe containing reference training data)
    
    '''
    
    if data == 'ligand dose response only':
        df_ref_1 = pd.DataFrame(data_, columns = ['L'])
        df_ref = pd.concat([df_ref_1], axis=1)
    
    elif data == 'ligand dose response and DBD dose response':
        df_ref_1 = pd.DataFrame(data_[:11], columns = ['L'])
        df_ref_2 = pd.DataFrame(data_[11:19], columns = ['DBD_20'])
        df_ref_3 = pd.DataFrame(data_[19:], columns = ['DBD_10'])
        df_ref = pd.concat([df_ref_1, df_ref_2, df_ref_3], axis=1)
        
    #Save results
    filename = 'REFERENCE TRAINING DATA.xlsx'

    with pd.ExcelWriter(filename) as writer:  # doctest: +SKIP
        df_ref.to_excel(writer, sheet_name = 'Ref')
    
    return df_ref


def generateRefData(p_ref):
    
    ''' 
    Purpose: Generate reference training data (with and without noise) based on parameter set, p_ref
        
    Inputs: 
        p_ref: list of floats, each float corresponds to a reference parameter value 
                (length = # free parameters). Parameter labels defined in init() 
                (in Settings.py) and SolveAll() (in Run.py).
           
    Output: None
    
    Figures: TRAINING DATA.svg (plot showing reference training data)
    '''
    
    os.chdir(full_path)
    sub_folder_name = './REFERENCE DATA'
    createFolder('./' + sub_folder_name)
    os.chdir('./' + sub_folder_name)
    
    #Solve for simulation data
    doses, norm_solutions, chi2 = solveAll(p_ref)
    
    if data == 'ligand dose response only':
        #Add technical error
        noise_rep = addNoise(norm_solutions, 0)   
        noise_rep = [i/max(noise_rep) for i in noise_rep]

        #Plot true trajectory and values with noise
        fig = plt.figure(figsize = (3,3))
        ax1 = plt.subplot(111)   
        ax1.plot(doses,  norm_solutions, color = 'black', marker = None, 
                 linestyle = ':', label = 'true')
        ax1.errorbar(doses, noise_rep, color = 'black', marker = 'o', 
                     yerr = error, markerSize = 6, fillstyle = 'none', 
                     linestyle = 'none',capsize = 2, label = 'noise')
        ax1.set_xscale('symlog')
        ax1.set_xlabel('Ligand (nM)')
        ax1.set_ylabel('Reporter expression')
        plt.savefig('./REFERENCE TRAINING DATA' + '.svg')

    elif data == 'ligand dose response and DBD dose response':
        #Unpack simulations
        norm_solutions_ligand = norm_solutions[:11]
        norm_solutions_dbd_20 = norm_solutions[11:19]
        norm_solutions_dbd_10 = norm_solutions[19:]
        solutions_dbd_norm = [norm_solutions_dbd_20, norm_solutions_dbd_10]
        
        #Add technical error
        noise_solutions = addNoise(norm_solutions, 0)
        noise_solutions_ligand = noise_solutions[:11]
        noise_solutions_dbd_20 = noise_solutions[11:19]
        noise_solutions_dbd_10 = noise_solutions[19:]
        solutions_dbd_noise = [noise_solutions_dbd_20, noise_solutions_dbd_10]

        #Plot 
        fig = plt.figure(figsize = (6,3))
        fig.subplots_adjust(wspace=0.2)
        ax1 = plt.subplot(121)   
        ax2 = plt.subplot(122)

        #Ligand dose response (true model trajectory and training data with noise)`
        error_ = error[:11]
        ax1.plot(doses_ligand,  norm_solutions_ligand, color = 'black', marker = None, 
                 linestyle = ':', label = 'ref')
        ax1.errorbar(doses_ligand,  noise_solutions_ligand, color = 'black', marker = 'o', 
                     yerr = error_, markerSize = 6, fillstyle = 'none', linestyle = 'none',
                     capsize = 2, label = 'td')
        ax1.set_xscale('symlog')
        ax1.set_xlabel('Ligand (nM)')
        ax1.set_ylabel('Reporter expression')
        ax1.legend(loc = 'upper left')
    
        #DBD dose response 
        linestyles = [':', ':']
        labels = ['20ngAD', '10ngAD']
        colors = ['black', 'grey']
        error_ = error[11:19]
        for i in [0, 1]:
            ax2.plot(doses_dbd, solutions_dbd_norm[i], linestyle = linestyles[i], 
                     label = labels[i], color = colors[i])
            ax2.errorbar(doses_dbd, solutions_dbd_noise[i], color = colors[i], 
                         marker = 'o', yerr = error_, markerSize = 6, fillstyle = 'none', 
                         linestyle = 'none',capsize = 2)
        ax2.set_xlabel('DBD plasmid (ng)')
        ax2.set_ylabel('Reporter expression')
        ax2.legend()
        plt.show()
        plt.savefig('./REFERENCE TRAINING DATA.svg', dpi = 600)
        
    #Calculate chi2 between reference training data with and without noise
    chi2 = calcChi2(noise_solutions, norm_solutions, error)
    print('chi2(p_ref) = ' + str(round(chi2, 4)))
    
    #Save reference dataframe as an excel sheet
    saveRefData(noise_solutions)

p_ref = [15, 0.05, .047, 36, 100, 2]    
#generateRefData()


def testSingleSet():
    ''' 
    Purpose: Simulate the entire dataset for a single parameter set, p
        
    Inputs: 
        p: list of floats, each float corresponds to a parameter value 
            (length = # free parameters). Parameter labels defined in init() (in Settings.py) 
             and SolveAll() (in Run.py).
           
    Output: None
    
    Figures: FIT TO TRAINING DATA.svg 
        (plot showing experimental and simulated data on the same axes)
    
    '''
    os.chdir(full_path)
    sub_folder_name = './TEST'
    createFolder('./' + sub_folder_name)
    os.chdir('./' + sub_folder_name)
    
    doses, solutions, chi2 = solveAll(p, exp_data)
    R_sq = calcRsq(solutions, exp_data)  
    print('*******')
    print('R2: ' + str(np.round(R_sq, 3)))
    print('chi2: ' + str(np.round(chi2, 3)))
    print('*******')
 
    exp_ligand = exp_data[:11]
    exp_dbd_20 = exp_data[11:19]
    exp_dbd_10 = exp_data[19:]
   
    error_ligand = error[:11]
    error_dbd_20 = error[11:19]
    error_dbd_10 = error[19:]
    
    if data == 'ligand dose response only':
        #Plot true trajectory and values with noise
        fig = plt.figure(figsize = (3,3))
        ax1 = plt.subplot(111)   
        ax1.plot(doses_ligand, solutions, color = 'black', marker = None, 
                 linestyle = ':', label = 'sim')
        ax1.errorbar(doses_ligand, exp_ligand , color = 'black', marker = 'o', 
                     yerr = error_ligand, markerSize = 6, fillstyle = 'none', 
                     linestyle = 'none',capsize = 2, label = 'TD')
        ax1.set_xscale('symlog')
        ax1.set_xlabel('LIGAND DOSE (NM)')
        ax1.set_ylabel('REPORTER EXPRESSION')
        plt.savefig('./FIT TO TRAINING DATA.svg')     

    elif data == 'ligand dose response and DBD dose response': 
        #Unpack simulation results
        solutions_ligand = solutions[:11]
        solutions_dbd_20 = solutions[11:19]
        solutions_dbd_10 = solutions[19:]
        
        solutions_dbd_norm = [solutions_dbd_20, solutions_dbd_10]
        solutions_dbd_exp = [exp_dbd_20, exp_dbd_10]
        solutions_dbd_error = [error_dbd_20, error_dbd_10]
    
        #Plot 
        fig = plt.figure(figsize = (6,3))
        fig.subplots_adjust(wspace=0.2)
        ax1 = plt.subplot(121)   
        ax2 = plt.subplot(122)
        
        #Ligand dose response
        ax1.plot(doses_ligand, solutions_ligand, color = 'black', marker = None, 
                 linestyle = ':', label = 'sim')
        ax1.errorbar(doses_ligand, exp_ligand, color = 'black', marker = 'o', 
                     yerr = error_ligand, markerSize = 6, fillstyle = 'none', 
                     linestyle = 'none',capsize = 2, label = 'TD')
        ax1.set_xscale('symlog')
        ax1.set_xlabel('Rapalog (nM)')
        ax1.set_ylabel('Reporter expressio')
        ax1.legend(loc = 'upper left')
    
        #DBD dose response 
        linestyles = [':', ':']
        labels = ['20ng AD', '10ng AD']
        colors = ['black', 'grey']
        for i in [0, 1]:
            ax2.plot(doses_dbd, solutions_dbd_norm[i], linestyle = linestyles[i], label = labels[i], 
                     color = colors[i])
            ax2.errorbar(doses_dbd, solutions_dbd_exp[i], color = colors[i], marker = 'o', 
                         yerr = solutions_dbd_error[i], markerSize = 6, fillstyle = 'none', 
                         linestyle = 'none',capsize = 2)
        ax2.set_ylabel('Reporter expression')
        ax2.set_xlabel('FKBP-ZF plasmid (ng)')
        ax2.legend()
        plt.show()
        plt.savefig('./FIT TO TRAINING DATA.svg', dpi = 600)
        
    #Plot timecourse at 100 nM Ligand
    tp1_ = 18
    tp2_ = 24
    [e, b, k_bind, m, km, n] = p
    v = [[], []]
    v[0] = [50, 50, 100] #set base case doses
    v[1] = p
    v[0][2] = 100 * e
    output = 'timecourse'
    args =  [0, v, tp1_, tp2_,  output, model]
    solveSingle(args)
    
#p = [14.50971165,	0.001421422,	0.462093372,	3.080760271,	354.9922971,	1.40856275]
#testSingleSet()


    
