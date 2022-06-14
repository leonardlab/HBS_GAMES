#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:48:43 2022

@author: kate
"""
import cycler
from math import log10
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from config import Settings, ExperimentalData
plt.style.use("./paper.mplstyle.py")

def plot_x_y(data=list, labels=list, filename=str, x_scale=str) -> None:
    """Plots a 2-dimensional figure.

    Parameters
    ----------
    data
        list of lists containing the data (both experimental and simulated)
        data = [x, y_sim, y_exp, y_exp_error]

        x_values
            list of floats defining the independent variable

        y_sim
            list of floats defining the simulated dependent variable

        y_exp
            list of floats defining the experimental dependent variable

        y_exp_error
            list of floats defining the experimental error for the dependent variable

    labels
        list of strings defining the plot labels

        x_label
           string defining the label for the independent variable

        y_label
           string defining the label for the dependent variable

    filename
       string defining the filename used to save the plot

    x_scale
       string defining the scale for the independent variable

    Returns
    -------
    None

    """

    [x_values, y_sim, y_exp, y_exp_error] = data
    [x_label, y_label] = labels

    if ExperimentalData.data_type == "PEM evaluation":
        color_ = "dimgrey"
        marker_ = "^"
    else:
        color_ = "black"
        marker_ = "o"

    plt.figure(figsize=(3, 3))
    plt.plot(x_values, y_sim, linestyle="dotted", marker="None", label="sim", color=color_)
    if y_exp != "None":
        plt.errorbar(
            x_values,
            y_exp,
            marker=marker_,
            yerr=y_exp_error,
            color=color_,
            ecolor=color_,
            markersize=6,
            fillstyle="none",
            linestyle="none",
            capsize=2,
            label="exp",
        )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xscale(x_scale)
    plt.legend()
    plt.savefig("./" + filename + ".svg", dpi=600)

def plot_parameter_distributions_after_optimization(df: pd.DataFrame) -> None:
    """Plot parameter distributions across initial guesses 
     following parameter estimation with training data 

    Parameters
    ----------
    df
        a dataframe containing the parameter estimation results after optimization

    Returns
    -------
    None
    
    Figures
    -------
    './FITS r_sq ABOVE 0.99.svg' (plot of training data and simulated data for 
         parameter sets with r_sq > = 0.99)
    'OPTIMIZED PARAMETER DISTRIBUTIONS.svg' (plot of parameter distributions for 
         parameter sets with r_sq > = 0.99)
   
    """

    #Only keep rows for which r_sq >= .99
    df = df[df["r_sq"] >= 0.99]
    dose_responses = list(df['Simulation results'])
    
    # =============================================================================
    # 1. dose response for parameter sets with r_sq > .99
    # ============================================================================
    fig = plt.figure(figsize = (3,3))
    ax1 = plt.subplot(111)  

    #Plot experimental/training data
    ax1.errorbar(ExperimentalData.x, ExperimentalData.exp_data, color = 'black', marker = 'o', yerr = ExperimentalData.exp_error,  
                 fillstyle = 'none', linestyle = 'none',capsize = 2, label = 'Training data')
    ax1.set_xscale('symlog')
    
    #Plot simulated data for each parameter set in df
    n = len(dose_responses)
    color = plt.cm.Blues(np.linspace(.1, 1, n))
    plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    count = 0
    for dose_response in dose_responses:
        count += 1
        ax1.plot(ExperimentalData.x, dose_response, linestyle = ':', marker = None, 
                 label = 'Model fit ' + str(count))
    ax1.set_xlabel('Ligand dose (nM)')
    ax1.set_ylabel('Reporter expression')
    plt.savefig('./FITS R_SQ ABOVE 0.99.svg', bbox_inches="tight")
    
    # =============================================================================
    # 2. parameter distributions for parameter sets with r_sq > .99
    # =============================================================================
    df_log = pd.DataFrame()
    for label in Settings.parameter_labels:
        label_star = label + '*'
        new_list = [log10(i) for i in list(df[label_star])]
        df_log[label] = new_list
    df_log['r_sq'] = df['r_sq']
  
    plt.subplots(1,1, figsize=(4,3), sharex = True)
    df_log = pd.melt(df_log, id_vars=['r_sq'], value_vars=Settings.parameter_labels)
    ax = sns.boxplot(x='variable', y='value', data=df_log, color = 'dodgerblue')
    ax = sns.swarmplot(x='variable', y='value', data=df_log, color="gray")
    ax.set(xlabel='Parameter', ylabel='log(value)')
    plt.savefig('OPTIMIZED PARAMETER DISTRIBUTIONS.svg', dpi = 600)
