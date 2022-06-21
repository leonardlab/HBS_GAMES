#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:17:53 2022

@author: kate
"""
from typing import Tuple
import matplotlib.pyplot as plt
from games.config.settings import settings
from games.config.experimental_data import ExperimentalData

plt.style.use(settings["context"] + "paper.mplstyle.py")


def define_training_data_plot_settings() -> Tuple[str, str, str]:
    """Defines plot settings based on the dataID and data_type

    Parameters
    ----------
    None

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

    if ExperimentalData.data_type == "PEM evaluation":
        plot_color = "dimgrey"
        marker_type = "^"

    else:
        plot_color = "black"
        marker_type = "o"

    return x_label, y_label, x_scale, plot_color, marker_type


def plot_training_data(
    x_values: list, y_sim: list, y_exp: list, y_exp_error: list, filename: str
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

    Returns
    -------
    None
    """
    x_label, y_label, x_scale, plot_color, marker_type = define_training_data_plot_settings()
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
    plt.legend()
    plt.savefig("./" + filename + ".svg", dpi=600)
