#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:11:29 2022

@author: kate
"""

import math
from typing import Tuple, List
import numpy as np
from scipy.integrate import odeint
from games.plots.plots_training_data import plot_training_data_2d


class HBS_model:
    """
    Representation of HBS model

    """
##### START here#######
#include all model versions in single class, use ID to set state labels, ODEs, params
    def __init__(
        self,
        parameters: List[float] = [1, 1, 1, 1, 1, 1],
        inputs: List[float] = None,
        input_ligand: float = 1000,
        mechanismID: str = "default",
    ) -> None:

        """Initializes HBS model.

        Parameters
        ----------
        parameters
            List of floats defining the parameters

        inputs
            List of floats defining the inputs

        # input_ligand
        #     Float defining the input ligand concentration

        mechanismID
            a string defining the mechanism identity

        Returns
        -------
        None

        """
        self.state_labels = [
            "ZF mRNA",
            "ZF protein",
            "AD mRNA",
            "AD protein",
            "Ligand",
            "Activator",
            "Rep RNA",
            "Rep protein",
        ]
        self.parameters = parameters
        self.inputs = inputs
        self.input_ligand = input_ligand
        self.mechanismID = mechanismID
        number_of_states = len(self.state_labels)
        y_init = np.zeros(number_of_states)
        self.initial_conditions = y_init

    def solve_single(
        self, parameter_labels: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Solves synTF_Chem model for a single set of parameters and inputs, including 2 steps
           1) Time from transfection to ligand addition
           2) Time from ligand addition to measurement via flow cytometry

        Parameters
        ----------
        parameter_labels
            a list of strings defining the parameter labels

        Returns
        -------
        solution
            An array of ODE solutions (rows are timepoints and columns are model states)

        t
            A 1D array of time values corresponding to the rows in solution

        """
        # solve before ligand addition
        timesteps = 100
        end_time1 = 18
        tspace_before_ligand_addition = np.linspace(0, end_time1, timesteps)
        t = tspace_before_ligand_addition
        solution_before_ligand_addition = odeint(
            self.gradient,
            self.initial_conditions,
            t,
            args=(
                self.parameters,
                self.inputs,
                self.mechanismID,
            ),
        )

        # solve after ligand addition
        for i, label in enumerate(parameter_labels):
            if label == "e":
                input_ligand_transformed = self.input_ligand * self.parameters[i]

        end_time2 = 24
        tspace_after_ligand_addition = np.linspace(0, end_time2, timesteps)
        t = tspace_after_ligand_addition
        initial_conditions_after_ligand_addition = np.array(solution_before_ligand_addition[-1, :])
        initial_conditions_after_ligand_addition[4] = input_ligand_transformed
        solution_after_ligand_addition = odeint(
            self.gradient,
            initial_conditions_after_ligand_addition,
            t,
            args=(
                self.parameters,
                self.inputs,
                self.mechanismID,
            ),
        )

        return (
            tspace_before_ligand_addition,
            tspace_after_ligand_addition,
            solution_before_ligand_addition,
            solution_after_ligand_addition,
        )

    @staticmethod
    def gradient(
        y: np.ndarray, t: np.ndarray, parameters: list, inputs: list, mechanismID: str
    ) -> np.ndarray:
        """Defines the gradient for synTF_Chem model.

        Parameters
        ----------
        y
            an array defining the initial conditions (necessary parameter to use ODEint to solve gradient)

        t
            an array defining the time (necessary parameter to use ODEint to solve gradient)

        parameters
            a list of floats defining the parameters

        inputs
            a list of floats defining the inputs

        mechanismID
            a string defining the mechanism identity


        Returns
        -------
        dydt
            a list of floats corresponding to the gradient of each model state at time t

        """

        [dose_a, dose_b] = inputs
        if mechanismID == "A" or mechanismID == "B":
            [_, b, k_bind, m, km, n] = parameters

        if mechanismID == "C" or mechanismID == "D":
            [_, b, k_bind, m_star, km, n] = parameters
            m = m_star * b

        fractional_activation_promoter = (b + m * (y[5] / km) ** n) / (
            1 + (y[5] / km) ** n + (y[1] / km) ** n
        )

        if math.isnan(fractional_activation_promoter):
            fractional_activation_promoter = 0

        k_txn = 1
        k_trans = 1
        kdeg_rna = 2.7
        kdeg_protein = 0.35
        kdeg_reporter = 0.029
        kdeg_ligand = 0.01

        dydt = np.array(
            [
                k_txn * dose_a - kdeg_rna * y[0],  # y0 A mRNA
                k_trans * y[0] - kdeg_protein * y[1] - k_bind * y[1] * y[3] * y[4],  # y1 A protein
                k_txn * dose_b - kdeg_rna * y[2],  # y2 B mRNA
                k_trans * y[2] - kdeg_protein * y[3] - k_bind * y[1] * y[3] * y[4],  # y3 B protein
                -k_bind * y[1] * y[3] * y[4] - y[4] * kdeg_ligand,  # y4 Ligand
                k_bind * y[1] * y[3] * y[4] - kdeg_protein * y[5],  # y5 Activator
                k_txn * fractional_activation_promoter - kdeg_rna * y[6],  # y6 Reporter mRNA
                k_trans * y[6] - kdeg_reporter * y[7],  # y7 Reporter protein
            ]
        )

        return dydt

    def solve_experiment(self, x: list, dataID: str, parameter_labels: List[str]) -> list:
        """Solve synTF_Chem model for a list of ligand values.

        Parameters
        ----------
        x
            a list of floats containing the independent variable

        dataID
            a string defining the dataID

        parameter_labels
            a list of strings defining the parameter labels

        Returns
        -------
        solutions
            A list of floats containing the value of the reporter protein
            at the final timepoint for each ligand amount

        """

        solutions = []

        if dataID in ("ligand dose response", "ligand dose response and DBD dose response"):
            self.inputs = [50, 50]  # ng
            for ligand in x[:11]:
                self.input_ligand = ligand
                _, _, _, sol = self.solve_single(parameter_labels)
                solutions.append(sol[-1, -1])

        if dataID == "ligand dose response and DBD dose response":
            for ad_dose in [20, 10]:  # ng
                for dbd_dose in x[11:19]:
                    self.inputs = [dbd_dose, ad_dose]
                    self.input_ligand = 100
                    _, _, _, sol = self.solve_single(parameter_labels)
                    solutions.append(sol[-1, -1])

        return solutions

    @staticmethod
    def normalize_data(solutions_raw: List[float], dataID: str) -> List[float]:
        """Normalizes data by maximum value

        Parameters
        ----------
        solutions_raw
            a list of floats defining the solutions before normalization

        dataID
            a string defining the dataID

        Returns
        -------
        solutions_norm
            a list of floats defining the dependent variable for the given
            dataset (after normalization)

        """

        if dataID == "ligand dose response and DBD dose response":
            # normalize ligand dose response
            solutions_norm_1 = [i / max(solutions_raw[:11]) for i in solutions_raw[:11]]

            # normalize DBD dose response
            solutions_norm_2 = [i / max(solutions_raw[11:]) for i in solutions_raw[11:]]

            # combine solutions
            solutions_norm = solutions_norm_1 + solutions_norm_2

        elif dataID == "ligand dose response":
            # normalize ligand dose response
            solutions_norm = [i / max(solutions_raw) for i in solutions_raw]

        return solutions_norm

    @staticmethod
    def plot_training_data(
        x: list,
        solutions_norm: List[float],
        exp_data: List[float],
        exp_error: List[float],
        filename: str,
        run_type: str,
        context: str,
        dataID: str,
    ) -> None:
        """
        Plots training data and simulated training data for a single parameter set

        Parameters
        ----------
        x
            list of floats defining the independent variable

        solutions_norm
            list of floats defining the simulated dependent variable

        exp_data
            list of floats defining the experimental dependent variable

        exp_error
            list of floats defining the experimental error for the dependent variable

        filename
           a string defining the filename used to save the plot

        run_type
            a string containing the data type ('PEM evaluation' or else)

        context
            a string defining the file structure context

        dataID
            a string defining the dataID

        Returns
        -------
        None"""

        # define plot settings
        if run_type == "default":
            plot_color = "black"
            marker_type = "o"

        elif run_type == "PEM evaluation":
            plot_color = "dimgrey"
            marker_type = "^"

        if dataID == "ligand dose response":
            y_label = "Rep. protein (au)"
            x_label = "Ligand (nM)"
            x_scale = "symlog"
            plot_settings = x_label, y_label, x_scale, plot_color, marker_type
            plot_training_data_2d(
                x, solutions_norm, exp_data, exp_error, filename, plot_settings, context
            )

        elif dataID == "ligand dose response and DBD dose response":
            # Define plot settings for ligand dose response
            y_label = "Rep. protein (au)"
            x_label = "Ligand (nM)"
            x_scale = "symlog"
            plot_settings = x_label, y_label, x_scale, plot_color, marker_type

            # plot ligand dose response
            filename_1 = filename + "ligand dose response"
            plot_training_data_2d(
                x[:11],
                solutions_norm[:11],
                exp_data[:11],
                exp_error[:11],
                filename_1,
                plot_settings,
                context,
            )

            # Define plot settings for DBD dose response
            y_label = "Rep. protein (au)"
            x_label = "DBD plasmid dose (ng)"
            x_scale = "linear"
            plot_settings = x_label, y_label, x_scale, plot_color, marker_type

            # Plot DBD dose response @ 20ng AD
            filename_2 = filename + "DBD dose response 20ng AD"
            plot_training_data_2d(
                x[11:19],
                solutions_norm[11:19],
                exp_data[11:19],
                exp_error[11:19],
                filename_2,
                plot_settings,
                context,
            )

            # Plot DBD dose response @ 10ng AD
            filename_3 = filename + "DBD dose response 10ng AD"
            plot_training_data_2d(
                x[19:],
                solutions_norm[19:],
                exp_data[19:],
                exp_error[19:],
                filename_3,
                plot_settings,
                context,
            )
