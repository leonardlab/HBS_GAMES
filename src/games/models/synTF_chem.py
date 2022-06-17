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

class synTF_chem:
    """
    Representation of synTF_chem model

    """

    def __init__(
        self,
        parameters=None,
        inputs=None,
        input_ligand=1000,
    ) -> None:
        
        """Initializes synTF_Chem model.

        Parameters
        ----------
        parameters
            List of floats defining the parame ters

        inputs
            List of floats defining the inputs

        input_ligand
            Float defining the input ligand concentration

        Returns
        -------
        None

        """
        self.state_labels = state_labels = ['ZF mRNA', 'ZF protein', 'AD mRNA', 'AD protein', 'Ligand', 'Rep RNA', 'Rep protein']
        self.parameters = np.array(parameters)
        self.inputs = np.array(inputs)
        self.input_ligand = input_ligand
        number_of_states = 8
        y_init = np.zeros(number_of_states)
        self.initial_conditions = y_init

    def solve_single(self) -> Tuple[np.ndarray, np.ndarray]:
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
        
        return tspace_before_ligand_addition, tspace_after_ligand_addition, solution_before_ligand_addition, solution_after_ligand_addition

    @staticmethod
    def gradient(y=np.ndarray, t=np.ndarray, parameters=list, inputs=list) -> np.ndarray:
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
        [_, b, k_bind, m, km, n] = parameters
        [dose_a, dose_b] = inputs

        fractional_activation_promoter = (b + m * (y[5] / km) ** n) / (
            1 + (y[5] / km) ** n + (y[1] / km) ** n
        )
    
        if math.isnan(fractional_activation_promoter):
            fractional_activation_promoter = 0
            
        K_TXN = 1
        K_TRANS = 1
        KDEG_RNA = 2.7
        KDEG_PROTEIN = 0.35
        KDEG_REPORTER = 0.029
        KDEG_LIGAND = 0.01
        
        dydt = np.array(
            [
                K_TXN * dose_a - KDEG_RNA * y[0],  # y0 A mRNA
                K_TRANS * y[0] - KDEG_PROTEIN * y[1] - k_bind * y[1] * y[3] * y[4],  # y1 A protein
                K_TXN * dose_b - KDEG_RNA * y[2],  # y2 B mRNA
                K_TRANS * y[2] - KDEG_PROTEIN * y[3] - k_bind * y[1] * y[3] * y[4],  # y3 B protein
                -k_bind * y[1] * y[3] * y[4] - y[4] * KDEG_LIGAND,  # y4 Ligand
                k_bind * y[1] * y[3] * y[4] - KDEG_PROTEIN * y[5],  # y5 Activator
                K_TXN * fractional_activation_promoter - KDEG_RNA * y[6],  # y6 Reporter mRNA
                K_TRANS * y[6] - KDEG_REPORTER * y[7],  # y7 Reporter protein
            ]
        )

        return dydt

    def solve_ligand_sweep(self, x_ligand=float) -> list:
        """Solve synTF_Chem model for a list of ligand values.

        Parameters
        ----------
        x_ligand
            A list of integers containing the ligand amounts to sweep over


        Returns
        -------
        solutions
            A list of floats containing the value of the reporter protein
            at the final timepoint for each ligand amount

        """

        solutions = []
        self.inputs = [50, 50]
        for ligand in x_ligand:
            self.input_ligand = ligand 
            _, _, _, sol = self.solve_single()
            solutions.append(sol[-1, -1])

        return solutions