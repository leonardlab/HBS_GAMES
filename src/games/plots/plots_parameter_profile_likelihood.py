#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:48:43 2022

@author: kate
"""

from math import log10
import matplotlib.pyplot as plt
from config.settings import settings
plt.style.use(settings["context"] + "paper.mplstyle.py")

def plot_chi_sq_distribution(chi_sq_distribution: list, threshold_chi_sq: float) -> None:
    """Plots threshold chi_sq value for PPL calculations

    Parameters
    ----------
    chi_sq_distribution
        a list of floats defining the chi_sq distribution used to calcualte the threshold
        
    threshold_chi_sq
        a float defining the threshold chi_sq value

    Returns
    -------
    None

    """

    plt.figure(figsize=(5, 3))
    plt.xlabel("chi_sq_ref - chi_sq_fit")
    plt.ylabel("Count")
    y, x, _ = plt.hist(chi_sq_distribution, bins=35, histtype="bar", color="dimgrey")
    plt.plot(
        [threshold_chi_sq, threshold_chi_sq],
        [0, max(y)],
        "-",
        lw=3,
        alpha=0.6,
        color="dodgerblue",
        linestyle=":",
    )
    plt.savefig("./chi_sq distribution", bbox_inches="tight", dpi=600)
    
def plot_parameter_profile_likelihood(parameter_label: str, calibrated_parameter_value: float, 
                                      fixed_parameter_values_both_directions: list, 
                                      chi_sq_PPL_list_both_directions: list,
                                      calibrated_chi_sq: float,
                                      threshold_chi_sq: float) -> None:
    '''
    Plots PPL results for a single parameter
    
    Parameters
    ---------- 
    parameter_label
        a string defining the parameter label

    calibrated_parameter_value
        a float containing the calibrated values for the given parameter

    fixed_parameter_values_both_directions
        a list of floats containing the values of the fixed parameter 
        (independent variable for PPL plot)
        
    chi_sq_PPL_list_both_directions
        a list of floats containing the chi_sq values
        (dependent variable for PPL plot)

    calibrated_chi_sq
        a float defining the chi_sq associated with the calibrated parameter set
        
    threshold_chi_sq
        a float defining the threshold chi_sq value
        
    Returns
    -------
    None
 
    '''
    #Restructure data
    calibrated_parameter_value_log = log10(calibrated_parameter_value)
    x = fixed_parameter_values_both_directions
    y = chi_sq_PPL_list_both_directions
    
    #Drop data points outside of bounds
    x_plot = []
    y_plot = []
    for j in range(0, len(x)):
        if abs(x[j]) > (10 ** -10):
            x_plot.append(x[j])
            y_plot.append(y[j])
        else:
            print('data point dropped - outside parameter bounds')
            print(x[j])
    x_log = [log10(i) for i in x_plot]
    
    #Plot PPL data
    fig = plt.figure(figsize = (3,3))
    plt.plot(x_log, y_plot, 'o-', color = 'black', markerSize = 4, fillstyle='none', zorder = 1)
    plt.xlabel(parameter_label)
    plt.ylabel('chi_sq')
    
    #Plot the calbrated value in blue
    x = [calibrated_parameter_value_log]
    y = [calibrated_chi_sq]
    plt.scatter(x, y, s=16, marker='o', color = 'dodgerblue', zorder = 2)
    plt.ylim([calibrated_chi_sq * .75, threshold_chi_sq * 1.2])
     
    #Plot threshold as dotted line
    x1 = [min(x_log), max(x_log)]
    y1 = [threshold_chi_sq, threshold_chi_sq]
    plt.plot(x1, y1, ':', color = 'dimgrey')
        
    plt.savefig('profile likelihood plot ' + parameter_label + '.svg')

def plot_parameter_profile_likelihood_consequences(df_profile_likelihood_results):
    pass
   