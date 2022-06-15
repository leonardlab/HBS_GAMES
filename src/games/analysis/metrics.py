#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:37:00 2022

@author: kate
"""
from sklearn.linear_model import LinearRegression
import numpy as np
from config import Settings

def calc_r_sq(data_x=list, data_y=list) -> float:

    """Calculate correlation coefficient, r_sq, between 2 datasets

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

    # Calculate r_sq
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

    chi_sq = 0
    for i, sim_val in enumerate(sim):  # for each datapoint
        err = ((exp_[i] - sim_val) / (std[i])) ** 2
        chi_sq = chi_sq + err

    return chi_sq