#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:17:53 2022

@author: kate
"""
from typing import List, Tuple
import matplotlib.pyplot as plt
from games.config.settings import settings

plt.style.use(settings["context"] + "paper.mplstyle.py")


def plot_training_data_2d(
    x_values: List[float],
    y_sim: List[float],
    y_exp: List[float],
    y_exp_error: List[float],
    filename: str,
    plot_settings: Tuple[str, str, str, str, str],
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

    plot_settings
        a list of strings defining the plot settings

    Returns
    -------
    None
    """
    x_label, y_label, x_scale, plot_color, marker_type = plot_settings
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
