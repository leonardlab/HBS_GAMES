#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:48:43 2022

@author: kate
"""
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from config import Context

plt.style.use("./paper.mplstyle.py")

class Plots:
    def plot_x_y(x, y_sim, y_exp, y_exp_error, x_label, y_label, filename, y_scale = 'linear'):
        """Plots a 2-dimensional figure.
  
        Parameters
        ----------
        x
            List of floats defining the independent variable
            
        y
            List of floats defining the dependent variable
        
        Returns
        -------
        None
        
        """
         
        fig = plt.figure(figsize = (3,3))
        plt.plot(x, y_sim, linestyle="dotted", marker="None", label = 'sim')
        plt.errorbar(x, y_exp, color = 'black', marker = 'o', 
                     yerr = y_exp_error, markersize = 6, fillstyle = 'none', 
                     linestyle = 'none', capsize = 2, label = 'exp')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.yscale(y_scale)
        plt.legend()
        plt.savefig(Context.folder_path + '/' + filename + '.svg', dpi = 600)
        
class CalculateMetrics:
    def calc_r_sq(data_x, data_y):
        """Calculate correlation coefficient, Rsq, between 2 datasets
        
        Parameters
        ----------
        dataX 
            list of floats - first set of data for comparison
            
        dataY 
            list of floats - second set of data for comparison
                
        Returns
        -------
        r_sq
            float - value of r_sq for dataX and dataY
        
        '"""
        
        #Restructure the data
        x = np.array(data_x)
        y = np.array(data_y)
        x = x.reshape((-1, 1))
        
        #Perform linear regression
        model_linear_regression = LinearRegression()
        model_linear_regression.fit(x,y)
        
        #Calculate Rsq
        r_sq = model_linear_regression.score(x, y)
       
        return r_sq

    def calc_chi_sq(exp, sim, std):
        """Calculate chi2 between 2 datasets with measurement error described by std
        
        Parameters
        ----------
        exp 
            experimental data (list of floats, length = # datapoints)
            
        sim
            simulated data (list of floats, length = # datapoints)
                            
        std
            measurement error for exp data (list of floats, length = # datapoints)
                
        Returns
        -------
        chi_sq
            chi2 value (float) 
        
        '"""
    
        #Initialize chi2
        chi_sq = 0
        
        #Calculate chi2
        for i, sim_val in enumerate(sim): #for each datapoint
            err = ((exp[i] - sim_val) / (std[i])) ** 2
            chi_sq = chi_sq + err
            
        return chi_sq
    
    