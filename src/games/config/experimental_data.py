#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:36:16 2022

@author: kate
"""
from typing import Tuple, List
import pandas as pd
from games.config.settings import settings


def define_experimental_data() -> Tuple[List[float], List[float], List[float]]:
    """ "
    Imports, normalizes, and defines experimental data

    Parameters
    ----------
    None

    Returns
    -------
    x
            a list of floats containing the values of the independent variable

    exp_data
            a list of floats containing the values of the dependent variable

    exp_error
            a list of floats containing the values of the measurement error
            for the dependent variable
    """

    x, exp_data_raw, exp_error_raw = import_data()
    exp_data, exp_error = normalize_data_by_maximum_value(
        exp_data_raw, settings["dataID"], exp_error_raw
    )

    return x, exp_data, exp_error


def import_data() -> Tuple[List[float], List[float], List[float]]:
    """Imports experimental data

       Parameters
       ----------
       None

       Returns
       -------
       x
       a list of floats defining the independent variable for the given dataset
    0.0
       exp_data_raw
       a list of floats defining the dependent variable for the given
       dataset (before normalization)

       exp_error_raw
        a list of floats defining the measurement error for
        the dependent variable for the given dataset (before normalization)
    """
    path = settings["context"] + "config/"
    filename = path + "training_data_" + settings["dataID"] + ".csv"
    df_exp = pd.read_csv(filename)
    x = list(df_exp["x"])
    exp_data_raw = list(df_exp["y"])
    exp_error_raw = list(df_exp["y_err"])

    return x, exp_data_raw, exp_error_raw


def normalize_data_by_maximum_value(
    solutions_raw: List[float], dataID: str, error_raw: List[float] = [0.0]
) -> Tuple[List[float], List[float]]:
    """Normalizes data by maximum value

    Parameters
    ----------
    solutions_raw
        a list of floats defining the solutions before normalization

    dataID
        a string defining the dataID

    error_raw
        a list of floats defining the measurement error before normalization
        default is 0 so that this function can also be used for simulated data

    Returns
    -------
    solutions_norm
    a list of floats defining the dependent variable for the given
    dataset (after normalization)

    error_norm
     a list of floats defining the measurement error for
     the dependent variable for the given dataset (after normalization),
     if relevant
    """

    if dataID == "ligand dose response and DBD dose response":
        # normalize ligand dose response
        solutions_norm_1 = [i / max(solutions_raw[:11]) for i in solutions_raw[:11]]

        # normalize DBD dose response
        solutions_norm_2 = [i / max(solutions_raw[11:]) for i in solutions_raw[11:]]
        solutions_norm = solutions_norm_1 + solutions_norm_2

        if len(error_raw) > 1:
            error_norm = [i / max(solutions_raw) for i in error_raw]
        else:
            error_norm = [0]

    else:
        solutions_norm = [i / max(solutions_raw) for i in solutions_raw]
        if len(error_raw) > 1:
            error_norm = [i / max(solutions_raw) for i in error_raw]
        else:
            error_norm = [0]

    return solutions_norm, error_norm
