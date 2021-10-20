#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 08:39:32 2021

@author: kate
"""

import pandas as pd
import matplotlib.pyplot as plt 
from Run import solveAll
from math import log10
import numpy as np
import cycler
import seaborn as sns

import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



real_param_labels_free = ['e', 'b', 'k_bind', 'm', 'km', 'n']
real_param_labels_all = ['e', 'b', 'k_bind', 'm', 'km', 'n']

from Run import exp_data

def plotPLConsequences(df, param_label):
    
    '''
    Purpose: Plot PPL results to evaluate consqeuences of unidentifiability 
            1)) parameter relationships and 
            2) internal states
    
    Inputs: 
        df: a dataframe containing the results of the PPL (from Calc_PL) for the given parameter
        param_label: a string defining the identity of the parameter for which the PPL 
            consequences are plotted
        
    Outputs: None
    
    Figures:
        'PARAMETER RELATIONSHIPS ALONG ' + param_label + '.svg' 
            (plot of parameter relationships for parameter = param_label)
        'INTERNAL STATES ALONG ' + param_label + '.svg' 
            (plot of internal model state dynamics for parameter = param_label)
        
    '''
    
    '''1. Plot parameter relationships'''
    fig = plt.figure(figsize = (3.5,4))
    
    x_ = list(df['fixed ' + param_label])
    x = [log10(val) for val in x_]
    y = list(df['fixed ' + param_label + ' all params'])
    
    def ConvertStringToList(string):
        string = string.replace('[', '')
        string = string.replace(']', '')
        li = list(string.split(", "))
        li = [float(i) for i in li]
        return li
    
    y = [ConvertStringToList(string_) for string_ in y]
    
    lists = []
    for i in range(0, len(real_param_labels_all)):
        for j in range(0, len(real_param_labels_free)):
            if real_param_labels_all[i] == real_param_labels_free[j]:
                lists.append([log10(y_[i]) for y_ in y])
            
    sns.set_palette('mako')
    
    m_vals = []
    b_vals = []
    kbind_vals = []
    for i, params in enumerate(y):
        m_vals.append(log10(params[3]))
        b_vals.append(log10(params[1]))
        kbind_vals.append(log10(params[2]))
    
    Z = m_vals
    Y = b_vals
    X = kbind_vals
        
    plotx,ploty, = np.meshgrid(np.linspace(np.min(X),np.max(X),len(m_vals)),\
                               np.linspace(np.min(Y),np.max(Y),len(m_vals)))
    plotz = interp.griddata((X,Y),Z,(plotx,ploty),method='linear')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(plotx,ploty,plotz,color = 'grey')  # or 'hot'
    
    ax.set_xlabel('kbind')
    ax.set_ylabel('b')
    ax.set_zlabel('m')
    
    plt.savefig('3D parameter relationships plot with kbind.svg', dpi = 600)
    # ratios = [log10(i) for i in ratios] 
    # plt.plot(x, ratios, linestyle = 'dotted', marker = 'o', markerSize = 4, 
    #               label = 'b/km', color = 'grey')
    
    # for j in range(0, len(real_param_labels_free)):
    #     plt.plot(x, lists[j], linestyle = 'dotted', marker = 'o', markerSize = 4, 
    #               label = real_param_labels_free[j])
    
    # plt.xlabel(param_label)
    # plt.ylabel('other parameters')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
    #             fancybox=True, shadow=False, ncol=3)
    # plt.xlim([-1.5, 3.5])
    # plt.savefig('PARAMETER RELATIONSHIPS ALONG ' + param_label + '.svg', dpi = 600)
    
   
    # '''2. Plot internal model states'''
    # n = len(y)
    # color = plt.cm.Blues(np.linspace(.2, 1,n))
    # plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    
    # fig, axs = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=False, figsize = (8, 4))
    # fig.subplots_adjust(hspace=.25)
    # fig.subplots_adjust(wspace=0.2)
    
    # for param_list in y: # for each parameter set
    #     t_1, t_2, solution_before, solution_on, state_labels = solveAll(param_list, exp_data)
        
    #     axs = axs.ravel()
    #     for i in range(0, len(state_labels)):
    #         axs[i].plot(t_1, solution_before[:,i], linestyle = 'dashed')
    #         axs[i].plot(t_2, solution_on[:,i])
            
    #         if i in [0, 4]:
    #             axs[i].set_ylabel('Simulation value (a.u.)', fontsize = 8)
    #         if i in [4,5,6,7]:
    #             axs[i].set_xlabel('Time (hours)', fontsize = 8)
            
    #         axs[i].set_title(state_labels[i], fontweight = 'bold', fontsize = 8)
            
    #     plt.savefig('INTERNAL STATES ALONG ' + param_label + '.svg', dpi = 600)
        
df = pd.read_excel('./Results/Results for GAMES manuscript/Model A 1000 + 100/MODULE 3 - PARAMETER IDENTIFIABILITY ANALYSIS/PROFILE LIKELIHOOD RESULTS m.xlsx')
plotPLConsequences(df, 'm')