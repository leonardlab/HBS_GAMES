#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:25:15 2020

@author: kate
"""

#Package imports
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#GAMES imports
from Saving import createFolder

#Import custom style file for plotting
plt.style.use('./paper.mplstyle.py')

dpi_ = 600
colors = ['teal', 'deeppink' ,'rebeccapurple', 'darkseagreen', 'darkorange', 'dimgrey', 
          'crimson', 'cornflowerblue', 'springgreen', 'sandybrown', 'lightseagreen', 'blue', 
          'palegreen', 'lightcoral', 'lightpink', 'lightsteelblue', 'indigo', 'darkolivegreen',
          'maroon', 'lightblue', 'gold', 'olive', 'silver', 'darkmagenta'] * 5 
     
def analyzeSingleRun(df_opt, count):
    
    '''
    Purpose: Analyze the results of a single PEM run and plot the CF trajectory for each 
             initial guess
    
    Inputs: 
        df_opt: a dataframe containing the optimization results
   
    Outputs: 
        list(df['chi2']): list of final, optimized chi2 values across initial guesses 
        list(df['Rsq']): list of final, optimized Rsq values across initial guesses 
    
    Figures:
        'CF TRAJECTORY PLOTS RUN ' + str(count) + '.svg' 
            (plot of CF vs function evaluations for each initial guess, count = run #)
        
    '''
  
    fig1 = plt.figure(figsize=(8,3))
    fig1.subplots_adjust(hspace=.25)
    fig1.subplots_adjust(wspace=0.3)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    
    df = df_opt
    num_ig = len(list(df['chi2_list']))
    fontsize_ = 12
 
    #Plot CF trajectory 
    for i in range(0, num_ig):
        holder = list(df['chi2_list'])
        chi2_list = holder[i]
        fxn_evals = np.arange(1, len(chi2_list) + 1)
       
        #Plot chi2 (CF) vs. fxn evaluation
        ax1.plot(fxn_evals, chi2_list, color = colors[i], label = 'ig ' + str(i + 1)) 
   
    ax1.set_title('CF tracking', fontweight = 'bold', fontsize = fontsize_ + 2)
    ax1.set_xlabel('Function evaluations', fontweight = 'bold', fontsize = fontsize_)
    ax1.set_ylabel('chi2', fontweight = 'bold', fontsize = fontsize_) 

    #Plot initial and final chi2
    initial_list = []
    final_list = []
    for i in range(0, num_ig):
        holder = list(df['chi2_list'])
        chi2_list = holder[i]
        initial_list.append(chi2_list[0])
        final_list.append(chi2_list[-1])
        fxn_evals = range(0, len(chi2_list))
    
    labels_ = list(range(1, 1 + num_ig))
    y0 = initial_list
    y = final_list
    
    x = np.arange(len(labels_))  # the label locations
    width = 0.35  # the width of the bars
    
    ax2.bar(x - width/2, y0, width, label='ig', color = 'black')
    ax2.bar(x + width/2,y, width, label='opt', color = colors)

    # Add xticks on the middle of the group bars
    ax2.set_xticks(x)
    ax2.set_xlabel('IG #', fontweight = 'bold', fontsize = fontsize_)
    ax2.set_ylabel('chi2', fontweight = 'bold', fontsize = fontsize_) 
    ax2.set_yscale('log')
    
    plt.savefig('CF TRAJECTORY PLOTS RUN ' + str(count)  + '.svg', dpi = dpi_)
    
    return list(df['chi2']),list(df['Rsq'])
  

def plotPemEvaluation(files, R_sq_pem_eval_pass, chi2_10_percent):
    
    '''
    Purpose: Evaluate and plot PEM evaluation criterion
   
    Inputs: 
        files: a list of dataframes containing the results for comparison
        R_sq_PEM_Eval_pass: a float defining the Rsq value for the PEM evaluation criterion
   
    Outputs: 
        None
    
    Figures: 
        'PEM EVALUATION CRITERION.svg' 
            (plot of optimized Rsq values for each PEM evaluation dataset, 
             shows PEM evaluation criterion)
        'PEM EVALUATION CRITERION R2 >= 0.90.svg' 
            (plot of optimized Rsq values for each PEM evaluation dataset, 
             shows PEM evaluation criterion only for parameter sets with R2 >=0.90)
    '''
    
    createFolder('./ANALYSIS')
    os.chdir('./ANALYSIS')
    
    run = []
    chi2_list = []
    R_sq_list = []
    min_cf_list = []
    max_R_sq_list = []
  
    for i, file in enumerate(files):
        chi2, Rsq = analyzeSingleRun(file, i + 1)
        chi2_list = chi2_list + chi2
        R_sq_list = R_sq_list + Rsq
        run_ = [i + 1] * len(Rsq)
        run = run  + run_
        
        val, idx = min((val, idx) for (idx, val) in enumerate(chi2))
        min_cf_list.append(val)
        max_R_sq = Rsq[idx]
        max_R_sq_list.append(max_R_sq)  
        
    df_all = pd.DataFrame(columns = ['run', 'chi2', 'Rsq'])
    df_all['run'] = run
    df_all['chi2'] = chi2_list
    df_all['Rsq'] = R_sq_list
    
    #only plot results for chi2 values in the bottom 10% of the CFs from the global search
    df_all = df_all[df_all['chi2'] <= chi2_10_percent]
    print('chi2_10_percent: ' + str(chi2_10_percent))

    #Plot PEM evaluation criterion
    plt.subplots(1,1, figsize=(4,3), sharex = True)
    ax = sns.boxplot(x='run', y='Rsq', data=df_all, color = 'white')
    ax = sns.swarmplot(x='run', y='Rsq', data=df_all, color="black")
    ax.set(xlabel='PEM evaluation dataset', ylabel='Rsq, opt', title = 'R2pass = ' + str(R_sq_pem_eval_pass))
    plt.savefig('PEM EVALUATION CRITERION.svg', dpi = dpi_)
    
    df_all=df_all[df_all.Rsq > 0.9]
    plt.subplots(1,1, figsize=(4,3), sharex = True)
    ax = sns.boxplot(x='run', y='Rsq', data=df_all, color = 'white')
    ax = sns.swarmplot(x='run', y='Rsq', data=df_all, color="black")
    ax.set(xlabel='PEM evaluation dataset', ylabel='Rsq, opt', title = 'R2pass = ' + str(R_sq_pem_eval_pass))
    plt.savefig('PEM EVALUATION CRITERION R2 >= 0.90.svg', dpi = dpi_)
    
    print('R2pass: '+ str(R_sq_pem_eval_pass))
  