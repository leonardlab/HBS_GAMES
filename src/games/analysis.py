#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:48:43 2022

@author: kate
"""
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from config import Settings

plt.style.use("./paper.mplstyle.py")


def plot_x_y(data=list, labels=list, filename=str, x_scale=str, color=str) -> None:
    """Plots a 2-dimensional figure.

    Parameters
    ----------
    data
        list of lists containing the data (both experimental and simulated)
        data = [x, y_sim, y_exp, y_exp_error]

        x_doses
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

    color
       string defining the plot color

    Returns
    -------
    None

    """

    [x_doses, y_sim, y_exp, y_exp_error] = data
    [x_label, y_label] = labels

    plt.figure(figsize=(3, 3))
    plt.plot(x_doses, y_sim, linestyle="dotted", marker="None", label="sim", color=color)
    if y_exp != "None":
        plt.errorbar(
            x_doses,
            y_exp,
            marker="o",
            yerr=y_exp_error,
            color=color,
            ecolor=color,
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


def calc_r_sq(data_x=list, data_y=list) -> float:

    """Calculate correlation coefficient, Rsq, between 2 datasets

    Parameters
    ----------
    data_x
        list of floats - first set of data for comparison

    data_y
        list of floats - second set of data for comparison

    Returns
    -------
    r_sq
        float - value of r_sq for dataX and dataY

    '"""

    # Restructure the data
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_x = data_x.reshape((-1, 1))

    # Perform linear regression
    model_linear_regression = LinearRegression()
    model_linear_regression.fit(data_x, data_y)

    # Calculate Rsq
    r_sq = model_linear_regression.score(data_x, data_y)

    return r_sq


def calc_chi_sq(exp_=list, sim=list, std=list) -> float:

    """Calculate chi2 between 2 datasets with measurement error described by std

    Parameters
    ----------
    exp_
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
    if Settings.weight_by_error == "no":
        std = [1] * len(exp_)

    # Initialize chi2
    chi_sq = 0

    # Calculate chi2
    for i, sim_val in enumerate(sim):  # for each datapoint
        err = ((exp_[i] - sim_val) / (std[i])) ** 2
        chi_sq = chi_sq + err

    return chi_sq
