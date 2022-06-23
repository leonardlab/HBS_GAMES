#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:17:53 2022

@author: kate
"""
from typing import Tuple
import matplotlib.pyplot as plt
from games.config.settings import settings

plt.style.use(settings["context"] + "paper.mplstyle.py")


def define_training_data_plot_settings(
    run_type: str, independent_variable: str = "default"
) -> Tuple[str, str, str, str, str]:
    """Defines plot settings based on the dataID and data_type

    Parameters
    ----------
    run_type
        a string containing the data type ('PEM evaluation' or else)

    independent_variable
        a string containing the name of the independent variable
        if the dataID has only a single dataset with a single independent variable,
        then this argument is unnecessary

    Returns
    -------
    x_label
        string defining the label for the independent variable

    y_label
        string defining the label for the dependent variable

    x_scale
        string defining the scale for the independent variable

    plot_color
        string defining the color for the plot

    marker_type
        string defining the marker type

    """
    y_label = "Rep. protein (au)"
    if settings["dataID"] == "ligand dose response":
        x_label = "Ligand (nM)"
        x_scale = "symlog"

    elif settings["dataID"] == "synTF dose response":
        x_label = "synTF (ng)"
        x_scale = "linear"

    elif settings["dataID"] == "ligand dose response and DBD dose response":
        if independent_variable == "ligand":
            x_label = "Ligand (nM)"
            x_scale = "symlog"
        elif independent_variable == "DBD":
            x_label = "DBD (ng)"
            x_scale = "linear"

    if run_type == "PEM evaluation":
        plot_color = "dimgrey"
        marker_type = "^"

    else:
        plot_color = "black"
        marker_type = "o"

    return x_label, y_label, x_scale, plot_color, marker_type


def plot_training_data_2d(
    x_values: list,
    y_sim: list,
    y_exp: list,
    y_exp_error: list,
    filename: str,
    run_type: str,
    independent_variable: str = "default",
) -> None:
    """Plots a 2-dimensional figure.

    Parameters
    ----------
    x_values
        list of floats defining the independent variable

    y_sim
        list of floats defining the simulated dependent variable

    y_exp
        list of floats defining the experimental dependent variable

    y_exp_error
        list of floats defining the experimental error for the dependent variable

    filename
       string defining the filename used to save the plot

    run_type
        a string containing the data type ('PEM evaluation' or else)

    independent_variable
        a string containing the name of the independent variable
        if the dataID has only a single dataset with a single independent variable,
        then this argument is unnecessary

    Returns
    -------
    None
    """
    x_label, y_label, x_scale, plot_color, marker_type = define_training_data_plot_settings(
        run_type, independent_variable
    )
    plt.figure(figsize=(3, 3))
    plt.plot(x_values, y_sim, linestyle="dotted", marker="None", label="sim", color=plot_color)
    plt.errorbar(
        x_values,
        y_exp,
        marker=marker_type,
        yerr=y_exp_error,
        color=plot_color,
        ecolor=plot_color,
        markersize=6,
        fillstyle="none",
        linestyle="none",
        capsize=2,
        label="exp",
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xscale(x_scale)
    plt.ylim([0, 1.15])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.legend()
    plt.savefig("./" + filename + ".svg", dpi=600)
