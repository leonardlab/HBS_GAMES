#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 2023

@author: Kathleen Dreyer
"""

import math
import numpy as np
from scipy.integrate import odeint
from games.plots.plots_training_data import plot_training_data_2d


class HBS_model:
    """
    Representation of HBS model

    """
    def __init__(
        self,
        parameters: list[float] = [1, 1, 1, 1, 1, 1],
        input_pO2: list[float] = None,
        mechanismID: str = "default",
        t_normoxia: np.ndarray = np.arange(0, 500, 1),
        t_hypoxia_exp: np.ndarray = np.array([0, 24, 48, 72, 96, 120]),
        t_hypoxia_plot: np.ndarray = np.linspace(0,120,31)
    ) -> None:

        """Initializes HBS model.

        Parameters
        ----------
        parameters
            List of floats defining the parameters

        input_pO2
            List of floats defining the input pO2: [pO2_normoxia, pO2_hypoxia]

        mechanismID
            a string defining the mechanism identity

        Returns
        -------
        None

        """
        self.parameters = parameters
        self.pO2_normoxia = input_pO2[0]
        self.pO2_hypoxia = input_pO2[1]
        self.mechanismID = mechanismID
        self.t_normoxia = t_normoxia
        self.t_hypoxia_exp = t_hypoxia_exp
        self.t_hypoxia_plot = t_hypoxia_plot

        if self.mechanismID == "A":
            self.state_labels = [
                'HAFR', 'HAFP', 'aHIF', 'HIF1R', 'HIF1P', 
                'HIF2R', 'HIF2P', 'HIF2P*', 'DSRE2R', 'DSRE2P'
            ]
            # self.topology_gradient = self.topology_gradient_A
        
        elif self.mechanismID == "B" or self.mechanismID == "B2":
            self.state_labels = [
                'HAFR', 'HAFP', 'aHIF', 'HIF1R', 'HIF1P', 
                'HIF2R', 'HIF2P', 'HIF2P*', 'DSRE2R', 'DSRE2P' 
            ]
            # self.topology_gradient = self.topology_gradient_B

        elif self.mechanismID == "C":
            self.state_labels = [
                'HAFR', 'HAFP', 'SUMOR', 'SUMOP', 'HAFS', 
                'aHIF', 'HIF1R', 'HIF1P', 'HIF2R', 'HIF2P',
                'HIF2P*', 'DSRE2R', 'DSRE2P'
            ]
            # self.topology_gradient = self.topology_gradient_C

        elif self.mechanismID == "D" or self.mechanismID == "D2":
            self.state_labels = [
                'HAFR', 'HAFP', 'SUMOR', 'SUMOP', 'HAFS', 
                'aHIF', 'HIF1R', 'HIF1P', 'HIF2R', 'HIF2P', 
                'HIF2P*', 'DSRE2R', 'DSRE2P'
            ]
            self.topology_gradient = self.topology_gradient_D

        number_of_states = len(self.state_labels)
        y_init = np.zeros(number_of_states)
        self.number_of_states = number_of_states
        self.initial_conditions = y_init

    def solve_single(
        self, topology: str, t_hypoxia: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Solves single HBS topology model for a single set of parameters and inputs,
        including 2 steps:
           1) time = 0 hours to time = 500 hours in normoxia (21% O2)
           2) time = 0 hours to time = 120 hours in hypoxia (1% O2)

        Parameters
        ----------
        parameter_labels
            a list of strings defining the parameter labels

        topology
            a string defining the topology ("simple" for simple HBS, "H1a_fb" for HBS
            with HIF1a feedback, or "H2a_fb" for HBS with HIF2a feedback)

        Returns
        -------
        solution
            An array of ODE solutions (rows are timepoints and columns are model states)

        t_hypoxia
            A 1D array of time values corresponding to the rows in solution

        """
        # solve in normoxia
        solution_normoxia = odeint(
            self.topology_gradient,
            self.initial_conditions,
            self.t_normoxia,
            args=(
                self.parameters,
                self.pO2_normoxia,
                self.mechanismID,
                topology
            ),
        )
        solution_normoxia_dict = {}
    
        for i in range(0,self.number_of_states):
            solution_normoxia_dict[self.state_labels[i]] = solution_normoxia[:,i]
        
        # solve in hypoxia
        initial_conditions_hypoxia = []
    
        for label in self.state_labels:
            initial_conditions_hypoxia.append(solution_normoxia_dict[label][-1])
            
        solution_hypoxia = odeint(
            self.topology_gradient,
            initial_conditions_hypoxia,
            t_hypoxia,
            args=(
                self.parameters,
                self.pO2_hypoxia,
                self.mechanismID,
                topology
            ),
        )
        solution_hypoxia_dict = {}

        for i in range(0,self.number_of_states):
            solution_hypoxia_dict[self.state_labels[i]] = solution_hypoxia[:,i]
        
        return solution_normoxia_dict, t_hypoxia, solution_hypoxia_dict

    @staticmethod
    def topology_gradient_D(
        y: np.ndarray, 
        t: np.ndarray, 
        parameters: list[float], 
        input_pO2: float, 
        mechanismID: str, 
        topology: str
    ) -> np.ndarray:
        """Defines the gradient for simple HBS topology model.

        Parameters
        ----------
        y
            an array defining the initial conditions (necessary parameter to use ODEint to solve gradient)

        t
            an array defining the time (necessary parameter to use ODEint to solve gradient)

        parameters
            a list of floats defining the parameters

        input_pO2
            a float defining the input O2 pressure (pO2)

        mechanismID
            a string defining the mechanism identity

        topology
            a string defining the topology ("simple" for simple HBS, "H1a_fb" for HBS with HIF1a feedback,
            or "H2a_fb" for HBS with HIF2a feedback)

        Returns
        -------
        dydt
            a list of floats corresponding to the gradient of each model state at time t

        """

        pO2 = input_pO2
        #mech ID statements
        if mechanismID == 'model_D2':
            ([
                t_HAF,
                k_txn2,
                k_dHAF, 
                k_bHS, 
                k_bHH, 
                k_txnH, 
                k_dH1R, 
                k_dH1P, 
                k_dHP, 
                k_txnBH
            ]) = parameters

        elif mechanismID == 'model_D':
            ([
                t_HAF,
                k_dHAF, 
                k_bHS, 
                k_bHH, 
                k_txnH, 
                k_dH1R, 
                k_dH1P, 
                k_dHP, 
                k_txnBH
            ]) = parameters
            
            k_txn2 = 1.0 #U
            
        #parameters that will be held constant:
        k_txn = 1.0 #U
        k_dR = 2.7 #1/h
        k_tln = 1 #U
        k_dP = 0.35 #1/h
        k_dRep = 0.029 #1/hr

        if pO2 == 138:
            k_dHAF = 0

        else:
            k_dHAF = np.piecewise(t, [t < t_HAF, t >= t_HAF], [k_dHAF, 0])

        dydt = np.array(
            [
            k_txn2 - k_dR*y[0], #y[0] = HAF mRNA
            k_tln*y[0] - k_dP*y[1] - k_dHAF*y[1] - (k_bHS/pO2)*y[1]*y[3], #y[1] = HAF protein
            k_txn - k_dR*y[2], # y[2] = SUMO mRNA
            k_tln*y[2] - k_dP*y[3] - (k_bHS/pO2)*y[1]*y[3], # y[3] = SUMO protein,
            (k_bHS/pO2)*y[1]*y[3] - k_dP*y[4] - k_bHH*y[9]*y[4], # y[4] = SUMO HAF
            k_txnH*(y[7] + y[10]) - k_dR*y[5], # y[5] = antisense HIF1a RNA
            k_txn - k_dR*y[6] - k_dH1R*y[5]*y[6], # y[6] = HIF1a mRNA
            k_tln*y[6] - k_dP*y[7] - k_dHP*pO2*y[7] - k_dH1P*y[7]*(y[1] + y[4]), # y[7] = HIF1a protein
            k_txn - k_dR*y[8], # y[8] = HIF2a mRNA
            k_tln*y[8] - k_dP*y[9] - k_dHP*pO2*y[9] - k_bHH*y[9]*y[4], # y[9] = HIF2a protein
            k_bHH*y[9]*y[4] - k_dP*y[10], # y[10] = HIF2a* protein
            k_txnBH*(y[7] + y[10]) - k_dR*y[11], # y[11] = DsRED2 mRNA
            k_tln*y[11] - k_dRep*y[12] # y[12] = DsRED2 protein
            ]
        )

        if topology == "simple":
            # no changes needed to dydt 
            return dydt
        elif topology == "H1a_fb":
            # y[6] = HIF1a mRNA, CHANGE dydt[6] for H1a fb
            dydt_H1a_fb = dydt.copy()
            dydt_H1a_fb[6] = k_txn + k_txnBH*(y[7] + y[10]) - k_dR*y[6] - k_dH1R*y[5]*y[6]
            return dydt_H1a_fb
        elif topology == "H2a_fb":
            # y[8] = HIF2a mRNA, CHANGE dydt[8] for H2a fb
            dydt_H2a_fb = dydt.copy()
            dydt_H2a_fb[8] = k_txn + k_txnBH*(y[7] + y[10]) - k_dR*y[8]
            return dydt_H2a_fb

    def solve_experiment(
        self, 
        t_hypoxia: np.ndarray, 
        topologies: list[str], 
        dataID: str
    ) -> dict:
        """Solve all HBS topology models in normoxia and hypoxia.

        Parameters
        ----------
        topologies
            a list of strings defining the topologies to be solved

        dataID
            a string defining the dataID

        Returns
        -------
        solutions
            A list of floats containing the value of the reporter protein
            at ????

        """
        all_topology_hypoxia_dict = {}

        for topology in topologies:
            _, _, solution_hypoxia_dict = self.solve_single(
                topology,
                t_hypoxia,
            )
            all_topology_hypoxia_dict[topology] = solution_hypoxia_dict

        return all_topology_hypoxia_dict

    # @staticmethod
    # def normalize_data(solutions_raw: dict[str, dict[str, float]], dataID: str) -> list[float]:
    #     """Normalizes data by maximum value

    #     Parameters
    #     ----------
    #     solutions_raw
    #         a list of floats defining the solutions before normalization

    #     dataID
    #         a string defining the dataID

    #     Returns
    #     -------
    #     solutions_norm
    #         a list of floats defining the dependent variable for the given
    #         dataset (after normalization)

    #     """
    #     if dataID == 'hypox only':
    #         DSRed2P_1a = np.append(SS_hox_1a[7.6]['DSRE2P'][:5], SS_hox_1a[138]['DSRE2P'][0])
    #         DSRed2P_4b = np.append(SS_hox_4b[7.6]['DSRE2P'], SS_hox_4b[138]['DSRE2P'][0])
    #         DSRed2P_4c = np.append(SS_hox_4c[7.6]['DSRE2P'], SS_hox_4c[138]['DSRE2P'][0])



    #     if dataID == "ligand dose response and DBD dose response":
    #         # normalize ligand dose response
    #         solutions_norm_1 = [i / max(solutions_raw[:11]) for i in solutions_raw[:11]]

    #         # normalize DBD dose response
    #         solutions_norm_2 = [i / max(solutions_raw[11:]) for i in solutions_raw[11:]]

    #         # combine solutions
    #         solutions_norm = solutions_norm_1 + solutions_norm_2

    #     elif dataID == "ligand dose response":
    #         # normalize ligand dose response
    #         solutions_norm = [i / max(solutions_raw) for i in solutions_raw]

    #     return solutions_norm

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
