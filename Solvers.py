#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 08:39:09 2020

@author: kate
"""

#Package imports
import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =============================================================================
# CODE TO CALCULATE COST FUNCTIONS
# ============================================================================= 
def calcRsq(data_x, data_y):
    ''' 
    Purpose: Calculate correlation coefficient, Rsq, between 2 datasets
        
    Inputs: 2 lists of floats (of the same length), dataX and dataY 
           
    Output: Rsq value (float) 
    
    '''
    
    #Restructure the data
    x = np.array(data_x)
    y = np.array(data_y)
    x = x.reshape((-1, 1))
    
    #Perform linear regression
    model = LinearRegression()
    model.fit(x,y)
    
    #Calculate Rsq
    Rsq = model.score(x, y)
   
    return Rsq

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

# =============================================================================
# CODE TO DEFINE ODES
# ============================================================================= 
def model_AB(y, t, v):
    ''' 
    Purpose: Define ODEs for reference model, call directly with odeint
        
    Inputs: 
        y: list of states
        t: list of timepoints
        v: list of model-specific arguments needed to define ODEs
           
    Output: 
        dydt: list of differential equations describing system
    
    '''
    
    #Unpack doses and parameters
    [[dose_a, dose_b, dose_l], [e, b, k_bind, m, km, n]] = v

    #Set fixed parameters
    k_txn = 1
    k_trans = 1
    kdeg_rna = 2.7  
    kdeg_protein = .35  
    kdeg_reporter = .029  
    k_deg_ligand = .01
    kdeg_a = kdeg_protein
    kdeg_b = kdeg_protein

    y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8 = y
    
    #Calculate fractical activation of promoter
    f_num = b + m * (y_6/km) ** n
    f_denom = 1 + (y_6/km) ** n + (y_2/km) ** n
    f = f_num/f_denom
    
    #If nan is present, set f to 0 
    if math.isnan(f):
        f = 0

    #Define ODEs
    dydt = [k_txn * dose_a - kdeg_rna * y_1, #y1 A mRNA
            k_trans * y_1 - kdeg_a * y_2 - k_bind * y_2 * y_4 * y_5, #y2 A protein
            k_txn * dose_b - kdeg_rna * y_3, #y3 B mRNA
            k_trans * y_3 - kdeg_b * y_4 - k_bind  * y_2 * y_4 * y_5, #y4 B protein
            - k_bind * y_2 * y_4 * y_5 - y_5 * k_deg_ligand, #y5 Ligand
            k_bind  * y_2 * y_4 * y_5 - kdeg_protein * y_6, #y6 Activator
            k_txn * f - kdeg_rna * y_7, #y7 Reporter mRNA
            k_trans * y_7 - kdeg_reporter * y_8] #y8 Reporter protein

    return dydt

def model_CD(y, t, v):
    ''' 
    Purpose: Define ODEs for reference model, call directly with odeint
        
    Inputs: 
        y: list of states
        t: list of timepoints
        v: list of model-specific arguments needed to define ODEs
           
    Output: 
        dydt: list of differential equations describing system
    
    '''
    
    #Unpack doses and parameters
    [[dose_a, dose_b, dose_l], [e, b, k_bind, m_star, km, n]] = v
    
    m = m_star
    b = 1

    #Set fixed parameters
    k_txn = 1
    k_trans = 1
    kdeg_rna = 2.7  
    kdeg_protein = .35  
    kdeg_reporter = .029  
    k_deg_ligand = .01 
    kdeg_a = kdeg_protein
    kdeg_b = kdeg_protein

    y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8 = y
    
    #Calculate fractical activation of promoter
    f_num = b + m * (y_6/km) ** n
    f_denom = 1 + (y_6/km) ** n + (y_2/km) ** n
    f = f_num/f_denom
    
    #If nan is present, set f to 0 
    if math.isnan(f):
        f = 0

    #Define ODEs
    dydt = [k_txn * dose_a - kdeg_rna * y_1, #y1 A mRNA
            k_trans * y_1 - kdeg_a * y_2 - k_bind * y_2 * y_4 * y_5, #y2 A protein
            k_txn * dose_b - kdeg_rna * y_3, #y3 B mRNA
            k_trans * y_3 - kdeg_b * y_4 - k_bind  * y_2 * y_4 * y_5, #y4 B protein
            - k_bind * y_2 * y_4 * y_5 - y_5 * k_deg_ligand, #y5 Ligand
            k_bind  * y_2 * y_4 * y_5 - kdeg_protein * y_6, #y6 Activator
            k_txn * f - kdeg_rna * y_7, #y7 Reporter mRNA
            k_trans * y_7 - kdeg_reporter * y_8] #y8 Reporter protein

    return dydt



# =============================================================================
# CODE TO SOLVE MODEL FOR A SINGLE DATA POINT
# ============================================================================= 
def solveSingle(args): 
    
    ''' 
    Purpose: Given a set of arguments (parameters, initial conditions), 
    solve the model for a single datapoint (for a single set of component doses)
        
    Inputs: 
        args (a list of arguments defining the condition) defined as
        args = [index, v, tp1_, tp2_,  output, model]
           
    Output: 
        if output == 'save internal states'
            t_1 (list of timepoints before ligand addition)
            t_2 (list of timepoints after ligand addition)
            solution_before (2D array, dynamic trajectory of all states before ligand addition)
            solution_on (2D array, dynamic trajectory of all states after ligand addition)
            state_labels (list of labels for each state variable in the model)
        
        OR 
        
        if output != 'save internal states'
            solution_on[-1, -1] (float, the value of the reporter state at the final timepoint)
    
    Figures: 
        if output == 'timecourse':
            './TIMECOURSE.svg' (plots of dynamic trajectories of each state variable)
    
    '''
    
    [index, v, tp1_, tp2_, output, model] = args

    #We only consider a single model structure in this work. 
    #To change the model structure (without changing the free params or training data)
    #Add an elif statement to this section of the code with the appropirate information. 
    
    num_states = 8
    output_state = 7
    state_labels = ['A mRNA', 'A protein', 'B mRNA', 'B protein', 
                    'Ligand', 'TF', 'Reporter mRNA', 'Reporter protein']
    
    if model in ['model A', 'model B']:
        ode_model = model_AB
        
    elif model in ['model C', 'model D']:
        ode_model = model_CD

    else: 
        print('Error: Model name does not exist.')

    #Define ligand dose
    dose_ligand = v[0][2]

    #Set time 
    numpoints = 100 #number of timepoints
    t_1 = [tp1_ * float(i) / (numpoints - 1) for i in range(numpoints)]  #range of timepoints
    
    #Set initial conditions to 0
    y0_before = [0] * num_states
    
    #1. Solve the ODEs for the time up until ligand addition
    solution_before = odeint(ode_model, y0_before, t_1, args=(v,), mxstep=50000000)
    
    #2. Solve equations at and after Ligand addition timepoint
    #set ligand initial condition to dose_ligand
    solution_before[-1, 4] =  dose_ligand
    
    #set initial conditions to the final timepoint values of 
    #each state variable from the "before" condition
    y0_on = solution_before[-1, :]

    #Set time
    numpoints = 25
    t_2_ = [tp2_ * float(i) / (numpoints - 1) for i in range(numpoints)]  # hours
  
    #Solve equations after Ligand addition 
    solution_on = odeint(ode_model, y0_on, t_2_, args=(v,), mxstep=500000000)
    
    #Define t_2 for plotting
    t_2 = [i + tp1_ for i in t_2_]  

    if output == 'timecourse':
        
        #Plot timecourse
        fig, axs = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=False, figsize = (8, 4))
        fig.subplots_adjust(hspace=.5)
        fig.subplots_adjust(wspace=0.3)
        
        axs = axs.ravel()
        for i in range(0, num_states):
            axs[i].plot(t_1, solution_before[:,i], color = 'black', linestyle = 'dashed')
            axs[i].plot(t_2, solution_on[:,i], color = 'black')
            axs[i].set_xlabel('Time (hours)', fontsize = 8)
            axs[i].set_ylabel('Simulation value (a.u.)', fontsize = 8)
            axs[i].set_title(state_labels[i], fontweight = 'bold', fontsize = 10)
            
            max1 = max(solution_before[:,i])
            max2 = max(solution_on[:,i])
            axs[i].set_ylim(top = max(max1, max2) + .1 * max(max1, max2) )
        plt.savefig('./TIMECOURSES.svg')
        
        return 'Timecourses saved as TIMECOURSES.svg'
              
    elif output == 'save internal states':
        return t_1, t_2, solution_before, solution_on, state_labels
    else:
        return solution_on[-1, output_state] 


    
