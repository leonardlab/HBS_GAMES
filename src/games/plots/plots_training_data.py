#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:17:53 2022

@author: kate
"""
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt


def plot_training_data_2d(
    y_sim: List[np.ndarray],
    y_exp: List[float],
    y_exp_error: List[float],
    filename: str,
    plot_settings: Tuple[str, str],
    context: str,
) -> None:
    """Plots a 2-dimensional figure.

    Parameters
    ----------

    y_sim
        a list of arrays defining the HBS simulation value for each
        topology (in the format [[simple HBS], [HBS with H1a feedback],
        [HBS with H2a feedback]])

    y_exp
        a list of floats defining the experimental relative DsRE2
        expression for each HBS topology (in the format
        [simple HBS values, HBS with H1a feedback values,
        HBS with H2a feedback values])

    y_exp_error
        a list of floats defining the experimental error for the
        relative DsRE2 expression for each HBS topology (in the
        format [simple HBS values, HBS with H1a feedback values,
        HBS with H2a feedback values])

    filename
       string defining the filename used to save the plot

    plot_settings
        a list of strings defining the plot settings

    context
        a string defining the absolute path to src/games

    Returns
    -------
    None
    """
    plt.style.use(context + "paper.mplstyle.py")
    plot_color1, plot_color2, marker_type = plot_settings

    t_experiment = [0.0, 24.0, 48.0, 72.0, 96.0, 120.0]
    t_simulation = np.linspace(0,120,31)

    fig = plt.figure(figsize = (6.6,2.6))
    fig.subplots_adjust(wspace=0.1)
    ax1 = plt.subplot(131)   
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    ax1.errorbar(
        t_experiment[:-1],
        y_exp[:5],
        marker=marker_type,
        yerr=y_exp_error[:5],
        color=plot_color1,
        ecolor=plot_color1,
        fillstyle="none",
        linestyle="none",
        capsize=2,
        label='1% O2 training data'
    )
    ax1.errorbar(
        t_experiment[0],
        y_exp[5],
        marker=marker_type,
        yerr=y_exp_error[5],
        color=plot_color2,
        ecolor=plot_color2,
        fillstyle="none",
        linestyle="none",
        capsize=2,
        label='21% O2 training data'
    )

    ax2.errorbar(
        t_experiment,
        y_exp[6:12],
        marker=marker_type,
        yerr=y_exp_error[6:12],
        color=plot_color1,
        ecolor=plot_color1,
        fillstyle="none",
        linestyle="none",
        capsize=2,
        label='1% O2 training data'
    )
    ax2.errorbar(
        t_experiment[0],
        y_exp[12],
        marker=marker_type,
        yerr=y_exp_error[12],
        color=plot_color2,
        ecolor=plot_color2,
        fillstyle="none",
        linestyle="none",
        capsize=2,
        label='21% O2 training data'
    )

    ax3.errorbar(
        t_experiment,
        y_exp[13:-1],
        marker=marker_type,
        yerr=y_exp_error[13:-1],
        color=plot_color1,
        ecolor=plot_color1,
        fillstyle="none",
        linestyle="none",
        capsize=2,
        label='1% O2 training data'
    )
    ax3.errorbar(
        t_experiment[0],
        y_exp[-1],
        marker=marker_type,
        yerr=y_exp_error[-1],
        color=plot_color2,
        ecolor=plot_color2,
        fillstyle="none",
        linestyle="none",
        capsize=2,
        label='21% O2 training data'
    )

    ax1.plot(
        t_simulation[:26],
        y_sim[0][:-1],
        linestyle="dotted",
        marker="None",
        color=plot_color1,
        label="1% O2 best fit"
    )
    ax1.plot(
        t_simulation[0],
        y_sim[0][-1],
        linestyle="dotted",
        marker="None",
        color=plot_color2
    )

    ax2.plot(
        t_simulation,
        y_sim[1][:-1],
        linestyle="dotted",
        marker="None",
        color=plot_color1,
        label="1% O2 best fit"
    )
    ax2.plot(
        t_simulation[0],
        y_sim[1][-1],
        linestyle="dotted",
        marker="None",
        color=plot_color2
    )

    ax3.plot(
        t_simulation,
        y_sim[2][:-1],
        linestyle="dotted",
        marker="None",
        color=plot_color1,
        label="1% O2 best fit"
    )
    ax3.plot(
        t_simulation[0],
        y_sim[2][-1],
        linestyle="dotted",
        marker="None",
        color=plot_color2
    )

    ax1.set_xlabel('Time Post-Plating (hours)')
    ax1.set_ylabel('Relative DsRE2 Expression')
    ax1.set_xticks([0, 20, 40, 60, 80, 100])
    ax1.set_ylim(ax2.get_ylim())
    ax1.set_title('Simple HBS')
    ax1.legend()
    ax1.set_box_aspect(1)
    
    ax2.set_xlabel('Time Post-Plating (hours)')
    ax2.set_ylabel('Relative DsRE2 Expression')
    ax2.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax2.set_title('HIF1a Feedback HBS')
    ax2.set_box_aspect(1)
    
    ax3.set_xlabel('Time Post-Plating (hours)')
    ax3.set_ylabel('Relative DsRE2 Expression')
    ax3.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax3.set_ylim(ax2.get_ylim())
    ax3.set_title('HIF2a Feedback HBS')
    ax3.set_box_aspect(1)

    plt.savefig("./" + filename + ".svg", dpi=600)
