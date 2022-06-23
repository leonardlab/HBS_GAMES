#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:11:29 2022

@author: kate
"""

import math
from typing import Tuple
import numpy as np
from scipy.integrate import odeint
from games.config.settings import settings
from games.plots.plots_training_data import plot_training_data_2d


class synTF_chem:
    """
    Representation of synTF_chem model

    """

    def __init__(
        self,
        parameters: list = None,
        inputs: list = None,
        input_ligand: float = 1000,
    ) -> None:

        """Initializes synTF_Chem model.

        Parameters
        ----------
        parameters
            List of floats defining the parameters

        inputs
            List of floats defining the inputs

        input_ligand
            Float defining the input ligand concentration

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
        self.parameters = np.array(parameters)
        self.inputs = np.array(inputs)
        self.input_ligand = input_ligand
        number_of_states = 8
        y_init = np.zeros(number_of_states)
        self.initial_conditions = y_init

    def solve_single(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Solves synTF_Chem model for a single set of parameters and inputs, including 2 steps
           1) Time from transfection to ligand addition
           2) Time from ligand addition to measurement via flow cytometry

        Parameters
        ----------
        None

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
            ),
        )

        # solve after ligand addition
        for i, label in enumerate(settings["parameter_labels"]):
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
            ),
        )

        return (
            tspace_before_ligand_addition,
            tspace_after_ligand_addition,
            solution_before_ligand_addition,
            solution_after_ligand_addition,
        )

    @staticmethod
    def gradient(y: np.ndarray, t: np.ndarray, parameters: list, inputs: list) -> np.ndarray:
        """Defines the gradient for synTF_Chem model.

        Parameters
        ----------
        parameters
            List of floats defining the parameters

        inputs
            List of floats defining the inputs


        Returns
        -------
        dydt
            An list of floats corresponding to the gradient of each model state at time t

        """

        [dose_a, dose_b] = inputs
        if settings["modelID"] == "A":
            [_, b, k_bind, m, km, n] = parameters

        if settings["modelID"] == "B":
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

    def solve_experiment(self, x: list, dataID: str) -> list:
        """Solve synTF_Chem model for a list of ligand values.

        Parameters
        ----------
        x
            a list of floats containing the independent variable

        dataID
            a string defining the dataID

        Returns
        -------
        solutions
            A list of floats containing the value of the reporter protein
            at the final timepoint for each ligand amount

        """

        solutions = []
        if (
            dataID == "ligand dose response"
            or dataID == "ligand dose response and DBD dose response"
        ):
            self.inputs = [50, 50]  # ng
            for ligand in x[:11]:
                self.input_ligand = ligand
                _, _, _, sol = self.solve_single()
                solutions.append(sol[-1, -1])

        if dataID == "ligand dose response and DBD dose response":
            for ad_dose in [20, 10]:  # ng
                for dbd_dose in x[11:19]:
                    self.inputs = [dbd_dose, ad_dose]
                    self.input_ligand = 100
                    _, _, _, sol = self.solve_single()
                    solutions.append(sol[-1, -1])

        return solutions

    @staticmethod
    def plot_training_data(
        x: list,
        solutions_norm: list,
        exp_data: list,
        exp_error: list,
        filename: str,
        run_type: str,
    ) -> None:
        """
        Plots training data and simulated training data

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

        Returns
        -------
        None"""
        if settings["dataID"] == "ligand dose response":
            plot_training_data_2d(x, solutions_norm, exp_data, exp_error, filename, run_type)

        elif settings["dataID"] == "ligand dose response and DBD dose response":
            filename_1 = filename + "ligand dose response"
            plot_training_data_2d(
                x[:11],
                solutions_norm[:11],
                exp_data[:11],
                exp_error[:11],
                filename_1,
                run_type,
                "ligand",
            )

            filename_2 = filename + "DBD dose response 20ng AD"
            plot_training_data_2d(
                x[11:19],
                solutions_norm[11:19],
                exp_data[11:19],
                exp_error[11:19],
                filename_2,
                run_type,
                "DBD",
            )

            filename_3 = filename + "DBD dose response 10ng AD"
            plot_training_data_2d(
                x[19:],
                solutions_norm[19:],
                exp_data[19:],
                exp_error[19:],
                filename_3,
                run_type,
                "DBD",
            )
