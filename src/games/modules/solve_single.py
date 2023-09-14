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
    x: list[float],
    exp_data: list[float],
    exp_error: list[float],
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

    all_topology_hypoxia_dict, normalization_value = model.solve_experiment(x)

    solutions_DsRE2P_simple = np.append(
        all_topology_hypoxia_dict["simple"][6.6]["DSRE2P"][:5],
        all_topology_hypoxia_dict["simple"][138.0]["DSRE2P"][0]
    )
    solutions_DsRE2P_H1a_fb = np.append(
        all_topology_hypoxia_dict["H1a_fb"][6.6]["DSRE2P"],
        all_topology_hypoxia_dict["H1a_fb"][138.0]["DSRE2P"][0]
    )
    solutions_DsRE2P_H2a_fb = np.append(
        all_topology_hypoxia_dict["H2a_fb"][6.6]["DSRE2P"],
        all_topology_hypoxia_dict["H2a_fb"][138.0]["DSRE2P"][0]
    )

    solutions = np.concatenate((
        solutions_DsRE2P_simple,
        solutions_DsRE2P_H1a_fb,
        solutions_DsRE2P_H2a_fb
    ))

    solutions_norm = model.normalize_data(solutions, normalization_value)
    chi_sq = calc_chi_sq(exp_data, solutions_norm, exp_error, weight_by_error)
    r_sq = calc_r_sq(exp_data, solutions_norm)

    return solutions_norm, chi_sq, r_sq


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
        settings["weight_by_error"]
    )

    filename = "fit to training data"
    run_type = "default"

    all_topology_hypoxia_dict, all_topology_DsRE2P = model.solve_experiment_for_plot(x)
    model.plot_training_data(
        all_topology_DsRE2P,
        exp_data,
        exp_error,
        filename,
        run_type,
        settings["context"]
    )

    plot_timecourses(all_topology_hypoxia_dict)

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

# settings = {
#    "context": "/Users/kdreyer/Documents/Github/HBS_GAMES2/src/games/",
#    "parameters": [0, 1, 0, 4.429493523, 0.198487505, 43.39121915, 9.527538436, 0.540226238, 1.079281817, 33.79767681],
#    "dataID" : "hypoxia_only"
# }
# model.parameters = settings["parameters"]
# x, exp_data, exp_error = define_experimental_data(
#    settings
# )
# solutions_norm, chi_sq, r_sq = solve_single_parameter_set(
#     x,
#     exp_data,
#     exp_error,
#     "no",
# )
# print(solutions_norm)
# print(chi_sq, r_sq)