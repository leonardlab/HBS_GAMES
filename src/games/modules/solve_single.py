#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 15:25:47 2022

@author: kate
"""
import os
import numpy as np
from games.models.set_model import model
from games.utilities.saving import create_folder
from games.utilities.metrics import calc_chi_sq, calc_r_sq
from games.plots.plots_timecourses import plot_timecourses
from games.config.experimental_data import define_experimental_data


def solve_single_parameter_set(
    t_hypoxia: np.ndarray,
    topologies: list[str], 
    exp_data: list[float],
    exp_error: list[float],
    dataID: str,
    weight_by_error: str,
) -> dict:
    """
    Solves model for a single parameter set

    Parameters
    ----------
    exp_data
        a list of floats containing the values of the dependent variable

    exp_error
        a list of floats containing the values of the measurement error
        for the dependent variable

    dataID
        a string defining the dataID

    weight_by_error
        a string defining whether the cost function should be weighted by error or not

    parameter_labels
        a list of strings defining the parameter labels

    Returns
    -------
    solutions_norm
        a list of floats containing the normalized simulation values
        corresponding to the dataID defined in Settings

    chi_sq
        a float defining the value of the cost function

    r_sq
        a float defining the value of the correlation coefficient (r_sq)
    """

    solutions_dict = model.solve_experiment(t_hypoxia, topologies, dataID)
    # solutions_norm = model.normalize_data(solutions_dict, dataID)
    # chi_sq = calc_chi_sq(exp_data, solutions_norm, exp_error, weight_by_error)
    # r_sq = calc_r_sq(exp_data, solutions_norm)

    return solutions_dict #, chi_sq, r_sq


def run_single_parameter_set(settings: dict, folder_path: str) -> tuple[list[float], float, float]:
    """Solves model for a single parameter set using dataID defined in settings["

    Parameters
    ----------
    settings
        a dictionary of run settings

    folder_path
        a string defining the path to the main results folder

    Returns
    -------
    solutions_norm
        a list of floats containing the normalized simulation
        values corresponding to the dataID defined in Settings

    chi_sq
        a float defining the value of the cost function

    r_sq
        a float defining the value of the correlation coefficient (r_sq)

    """
    sub_folder_name = "TEST SINGLE PARAMETER SET"
    path = create_folder(folder_path, sub_folder_name)
    os.chdir(path)
    model.parameters = settings["parameters"]
    x, exp_data, exp_error = define_experimental_data(settings)
    solutions_norm, chi_sq, r_sq = solve_single_parameter_set(
        x,
        exp_data,
        exp_error,
        settings["dataID"],
        settings["weight_by_error"],
        settings["parameter_labels"],
    )
    filename = "fit to training data"
    run_type = "default"
    plot_timecourses(settings["modelID"], settings["parameter_labels"])
    model.plot_training_data(
        x,
        solutions_norm,
        exp_data,
        exp_error,
        filename,
        run_type,
        settings["context"],
        settings["dataID"],
    )

    print("")
    print("*************************")
    print("Parameters")
    for i, label in enumerate(settings["parameter_labels"]):
        print(label + " = " + str(model.parameters[i]))
    print("")
    print("Metrics")
    print("R_sq = " + str(np.round(r_sq, 4)))
    print("chi_sq = " + str(np.round(chi_sq, 4)))
    print("*************************")

    return solutions_norm, chi_sq, r_sq



settings = {
   "context": "/Users/kdreyer/Desktop/Github/HBS_GAMES2/src/games/",
   "dataID": "hypoxia_only",
   "parameters": [47.1, 1.51, 0.324, 4.37, 1.08e-3, 22.7, 7.43e-4, 0.622, 0.593, 3.88]
}

model.parameters = settings["parameters"]

input_pO2, exp_data, exp_error = define_experimental_data(
   settings
)

solutions = solve_single_parameter_set(
    model.t_hypoxia_exp,
    ["simple", "H1a_fb", "H2a_fb"],
    exp_data,
    exp_error,
    settings["dataID"],
    "no",
)

# print(solutions)
for key, val in solutions.items():
    print(key, ": ")
    for sub_key, sub_val in val.items():
        print(sub_key, ": ", np.around(sub_val, 4))