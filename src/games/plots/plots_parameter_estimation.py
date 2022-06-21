#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:19:16 2022

@author: kate
"""
from math import log10
import cycler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from games.config.experimental_data import ExperimentalData
from games.config.settings import settings

plt.style.use(settings["context"] + "paper.mplstyle.py")


def plot_chi_sq_trajectory(chi_sq_list: list) -> None:
    """Plots the chi_sq trajectory following an optimization run

    Parameters
    ----------
    chi_sq_list
        a list of floats defining the chi_sq value at each function evaluation

    Returns
    -------
    None
    """
    plt.figure(figsize=(3, 3))
    plt.plot(
        range(0, len(chi_sq_list)),
        chi_sq_list,
        linestyle="dotted",
        marker="None",
        color="lightseagreen",
    )
    plt.xlabel("cost function evaluation")
    plt.ylabel("chi_sq")
    plt.savefig("chi_sq trajectory for best fit.svg", dpi=600)


def plot_parameter_distributions_after_optimization(df_opt: pd.DataFrame) -> None:
    """Plot parameter distributions across initial guesses
     following parameter estimation with training data

    Parameters
    ----------
    df_opt
        a dataframe containing the parameter estimation results after optimization

    Returns
    -------
    None

    Figures
    -------
    './FITS r_sq ABOVE 0.99.svg' (plot of training data and simulated data for
         parameter sets with r_sq > = 0.99)
    'OPTIMIZED PARAMETER DISTRIBUTIONS.svg' (plot of parameter distributions for
         parameter sets with r_sq > = 0.99)

    """

    # Only keep rows for which r_sq >= .99
    df_opt = df_opt[df_opt["r_sq"] >= 0.99]
    dose_responses = list(df_opt["Simulation results"])

    # =============================================================================
    # 1. dose response for parameter sets with r_sq > .99
    # ============================================================================
    fig = plt.figure(figsize=(3, 3))
    ax1 = plt.subplot(111)

    # Plot experimental/training data
    ax1.errorbar(
        ExperimentalData().x,
        ExperimentalData().exp_data,
        color="black",
        marker="o",
        yerr=ExperimentalData().exp_error,
        fillstyle="none",
        linestyle="none",
        capsize=2,
        label="Training data",
    )
    ax1.set_xscale("symlog")

    # Plot simulated data for each parameter set indf_opt
    n = len(dose_responses)
    color = plt.cm.Blues(np.linspace(0.1, 1, n))
    plt.rcParams["axes.prop_cycle"] = cycler.cycler("color", color)
    count = 0
    for dose_response in dose_responses:
        count += 1
        ax1.plot(
            ExperimentalData().x,
            dose_response,
            linestyle=":",
            marker=None,
            label="Model fit " + str(count),
        )
    ax1.set_xlabel("Ligand dose (nM)")
    ax1.set_ylabel("Reporter expression")
    plt.savefig("./fits r_sq above 0.99.svg", bbox_inches="tight")

    # =============================================================================
    # 2. parameter distributions for parameter sets with r_sq > .99
    # =============================================================================
    df_opt_log = pd.DataFrame()
    for label in settings["parameter_labels"]:
        label_star = label + "*"
        new_list = [log10(i) for i in list(df_opt[label_star])]
        df_opt_log[label] = new_list
    df_opt_log["r_sq"] = df_opt["r_sq"]

    plt.subplots(1, 1, figsize=(4, 3), sharex=True)
    df_opt_log = pd.melt(df_opt_log, id_vars=["r_sq"], value_vars=settings["parameter_labels"])
    ax = sns.boxplot(x="variable", y="value", data=df_opt_log, color="dodgerblue")
    ax = sns.swarmplot(x="variable", y="value", data=df_opt_log, color="gray")
    ax.set(xlabel="Parameter", ylabel="log(value)")
    plt.savefig("optimized parameter distributions.svg", dpi=600)
