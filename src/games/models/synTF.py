#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:11:29 2022

@author: kate
"""
from typing import Tuple, List
import numpy as np
from scipy.integrate import odeint
from games.plots.plots_training_data import plot_training_data_2d


class synTF:
    """
    Representation of synTF model

    """

    def __init__(
            self, 
            parameters: List[float] = None,
            inputs: List[float] = None
        ) -> None:
        """Initializes synTF model.

        Parameters
        ----------
        parameters
            List of floats defining the parameters

        inputs
            List of floats defining the inputs

        Returns
        -------
        None

        """
        self.state_labels = ["ZFa mRNA", "ZFa protein", "Rep RNA", "Rep protein"]
        self.parameters = parameters
        self.inputs = inputs
        number_of_states = len(self.state_labels)
        y_init = np.zeros(number_of_states)
        self.initial_conditions = y_init

    def solve_single(self) -> Tuple[np.ndarray, np.ndarray]:
        """Solves synTF model for a single set of parameters and inputs

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

        timesteps = 100
        end_time = 42
        tspace = np.linspace(0, end_time, timesteps)
        t = tspace
        solution = odeint(
            self.gradient,
            self.initial_conditions,
            t,
            args=(
                self.parameters,
                self.inputs,
            ),
        )

        return solution, t

    @staticmethod
    def gradient(
        y: np.ndarray, t: np.ndarray, parameters: List[float], inputs: List[float]
    ) -> np.ndarray:
        """Defines the gradient for synTF model.

        Parameters
        ----------
        y
            an array defining the initial conditions (necessary parameter to use ODEint to solve gradient)

        t
            an array defining the time (necessary parameter to use ODEint to solve gradient)

        parameters
            List of floats defining the parameters

        inputs
            List of floats defining the inputs

        Returns
        -------
        dydt
            An array corresponding to the gradient of each model state at time t

        """

        k_txn = 1
        k_trans = 1
        kdeg_rna = 2.7
        kdeg_protein = 0.35
        kdeg_reporter = 0.029

        [b, m, w] = parameters
        [dose_a] = inputs

        fractional_activation_promoter = b + m * w * y[1] / (1 + w * y[1])

        dydt = np.array(
            [
                k_txn * dose_a - kdeg_rna * y[0],  # y0 synTF mRNA
                k_trans * y[0] - kdeg_protein * y[1],  # y1 synTF protein
                k_txn * fractional_activation_promoter - kdeg_rna * y[2],  # y2 Reporter mRNA
                k_trans * y[2] - kdeg_reporter * y[3],  # y3 Reporter protein
            ]
        )

        return dydt

    def solve_experiment(self, x: List[float], dataID: str, parameter_labels) -> List[float]:
        """Solve synTF model for a list of synTF values.

        Parameters
        ----------
        x
            a list of floats containing the independent variable

        dataID
            a string defining the dataID

        parameter_labels
            a list of strings defining the parameter labels
            (variable not necessary for synTF, but is necessary for synTF_chem,
            so include default value here)

        Returns
        -------
        solutions
            A list of floats containing the value of the reporter protein
            at the final timepoint for each synTF amount

        """
        solutions = []
        if dataID == "synTF dose response":
            for synTF_amount in x:
                self.inputs = [synTF_amount]
                sol, _ = self.solve_single()
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
        if dataID == "synTF dose response":
            solutions_norm = [i / max(solutions_raw) for i in solutions_raw]

        return solutions_norm

    @staticmethod
    def plot_training_data(
        x: List[float],
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
            a list of floats defining the independent variable

        solutions_norm
            a list of floats defining the simulated dependent variable

        exp_data
            a list of floats defining the experimental dependent variable

        exp_error
            a list of floats defining the experimental error for the dependent variable

        filename
            a string defining the filename used to save the plot

        run_type
            a string containing the data type ('PEM evaluation' or else)

        context
            a string defining the absolute path to src/games

        dataID
            a string defining the data identity

        Returns
        -------
        None
        """
        # define plot settings
        x_label = "synTF (ng)"
        x_scale = "linear"
        y_label = "Rep. protein (au)"

        if run_type == "default":
            plot_color = "black"
            marker_type = "o"
        elif run_type == "PEM evaluation":
            plot_color = "dimgrey"
            marker_type = "^"

        plot_settings = x_label, y_label, x_scale, plot_color, marker_type

        # make plot
        plot_training_data_2d(
            x, solutions_norm, exp_data, exp_error, filename, plot_settings, context
        )
