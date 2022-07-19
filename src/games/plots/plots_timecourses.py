#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:48:46 2022

@author: kate
"""
from typing import List
import matplotlib.pyplot as plt
from games.models.set_model import model, settings

plt.style.use(settings["context"] + "paper.mplstyle.py")


def plot_timecourses(modelID: str, parameter_labels: List[str]) -> None:
    """Plots timecourses of internal states for a single set of inputs

    Parameters
    ----------
     modelID
        a string defining the modelID

     parameter_labels
        a list of strings defining the parameter labels

    Returns
    -------
    None

    """

    fig, axs = plt.subplots(
        nrows=2, ncols=int(len(model.state_labels) / 2), sharex=True, sharey=False, figsize=(8, 4)
    )
    fig.subplots_adjust(hspace=0.25)
    fig.subplots_adjust(wspace=0.2)
    axs = axs.ravel()
    if modelID == "synTF_chem":
        model.inputs = [50, 50]
        model.input_ligand = 1000
        (
            tspace_before_ligand_addition,
            tspace_after_ligand_addition,
            solution_before_ligand_addition,
            solution_after_ligand_addition,
        ) = model.solve_single(parameter_labels)
        tspace_after_ligand_addition = [
            i + max(tspace_before_ligand_addition) for i in list(tspace_after_ligand_addition)
        ]

        for i, label in enumerate(model.state_labels):
            axs[i].plot(
                tspace_before_ligand_addition,
                solution_before_ligand_addition[:, i],
                linestyle="dotted",
                marker="None",
                color="black",
            )
            axs[i].plot(
                tspace_after_ligand_addition,
                solution_after_ligand_addition[:, i],
                linestyle="solid",
                marker="None",
                color="black",
            )

            if i in [0, 4]:
                axs[i].set_ylabel("Simulation value (a.u.)", fontsize=8)
            if i in [4, 5, 6, 7]:
                axs[i].set_xlabel("Time (hours)", fontsize=8)

            axs[i].set_title(label, fontweight="bold", fontsize=8)

    elif modelID == "synTF":
        solution, t = model.solve_single()
        for i, label in enumerate(model.state_labels):
            axs[i].plot(t, solution, linestyle="dotted", marker="None", color="black")
            if i in [0, 2]:
                axs[i].set_ylabel("Simulation value (a.u.)", fontsize=8)
            if i in [2, 3]:
                axs[i].set_xlabel("Time (hours)", fontsize=8)
            axs[i].set_title(label[i], fontweight="bold", fontsize=8)

    plt.savefig("timecourses of internal model states.svg", dpi=600)
