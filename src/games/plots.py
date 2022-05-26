#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:48:43 2022

@author: kate
"""
import matplotlib.pyplot as plt

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
        plt.savefig('./' + filename + '.svg', dpi = 600)