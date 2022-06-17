#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:48:46 2022

@author: kate
"""
import matplotlib.pyplot as plt
from models.set_model import model
from config.settings import settings

def plot_timecourses() -> None:
    """Plots timecourses of internal states for a single set of inputs

    Parameters
    ----------
    None

    Returns
    -------
    None
    
    Files
    -------
    'timcourses of internal model states.svg'

    """
    fig, axs = plt.subplots(nrows=2, ncols=len(model.state_labels)/2, sharex=True, sharey=False, figsize = (8, 4))
    fig.subplots_adjust(hspace=.25)
    fig.subplots_adjust(wspace=0.2)
    axs = axs.ravel()
    if settings["modelID"] == "synTF_chem":
        model.inputs = [50, 50]
        model.input_ligand = 1000
        tspace_before_ligand_addition, tspace_after_ligand_addition, solution_before_ligand_addition, solution_after_ligand_addition = model.solve_single()
        tspace_after_ligand_addition = [i + max(tspace_before_ligand_addition) for i in list(tspace_after_ligand_addition)]
        
        for i in range(0, len(model.state_labels)):
            axs[i].plot(tspace_before_ligand_addition, solution_before_ligand_addition[:,i], linestyle = 'dotted', marker = 'None', color = 'black')
            axs[i].plot(tspace_after_ligand_addition, solution_after_ligand_addition[:,i], linestyle = 'solid', marker = 'None', color = 'black')
            
            if i in [0, 4]:
                axs[i].set_ylabel('Simulation value (a.u.)', fontsize = 8)
            if i in [4,5,6,7]:
                axs[i].set_xlabel('Time (hours)', fontsize = 8)
            
            axs[i].set_title(model.state_labels[i], fontweight = 'bold', fontsize = 8)
            
    elif settings["modelID"] == "synTF":
        solution, t = model.solve_single()
        for i in range(0, len(model.state_labels)):
            axs[i].plot(t, solution, linestyle = 'dotted', marker = 'None', color = 'black')
            if i in [0, 2]:
                axs[i].set_ylabel('Simulation value (a.u.)', fontsize = 8)
            if i in [2,3]:
                axs[i].set_xlabel('Time (hours)', fontsize = 8)
            axs[i].set_title(model.state_labels[i], fontweight = 'bold', fontsize = 8)
  
    plt.savefig('timcourses of internal model states.svg', dpi = 600)
    

    
    