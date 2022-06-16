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
from config.settings import settings

class synTF:
    """
    Representation of synTF model

    """

    def __init__(self, parameters=None, inputs=None) -> None:
        """Initializes synTF model.

        Parameters
        ----------
        free_parameters
            List of floats defining the free parameters

        inputs
            List of floats defining the inputs

        Returns
        -------
        None

        """

        self.parameters = np.array(parameters)
        self.inputs = np.array(inputs)
        number_of_states = 4
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
    def gradient(y=np.ndarray, t=np.ndarray, parameters=list, inputs=list) -> np.ndarray:
        """Defines the gradient for synTF model.

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
        
        K_TXN = 1
        K_TRANS = 1
        KDEG_RNA = 2.7
        KDEG_PROTEIN = 0.35
        KDEG_REPORTER = 0.029

        [g] = parameters
        [dose_a] = inputs

        dydt = np.array(
            [
                K_TXN * dose_a - KDEG_RNA * y[0],  # y0 synTF mRNA
                K_TRANS * y[0] - KDEG_PROTEIN * y[1],  # y1 synTF protein
                K_TXN * g * y[1] - KDEG_RNA * y[2],  # y2 Reporter mRNA
                K_TRANS * y[2] - KDEG_REPORTER * y[3],  # y3 Reporter protein
            ]
        )

        return dydt

    def solve_synTF_sweep(self, x=list) -> list:
        """Solve synTF model for a list of synTF values.

        Parameters
        ----------
        x
            list of conditions representing the input synTF amounts

        Returns
        -------
        synTF_amounts
            A list of integers containing the synTF amounts to sweep over

        solutions
            A list of floats containing the value of the reporter protein
            at the final timepoint for each synTF amount

        """
        solutions = []
        for synTF_amount in x:
            self.inputs = [synTF_amount]
            sol, _ = self.solve_single()
            solutions.append(sol[-1, -1])

        return solutions
