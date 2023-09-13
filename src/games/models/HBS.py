#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 2023

@author: Kathleen Dreyer
"""

import math
import numpy as np
from typing import List, Tuple
from scipy.integrate import odeint
from copy import deepcopy
from games.plots.plots_training_data import plot_training_data_2d

#x = [6.6, 138.0]
#t_normoxia: np.ndarray = np.arange(0, 500, 1),
# t_hypoxia_exp: np.ndarray = np.array([0, 24, 48, 72, 96, 120]),
# t_hypoxia_plot: np.ndarray = np.linspace(0,120,31)

class HBS_model:
    """
    Representation of HBS model

    """
    def __init__(
        self,
        parameters: List[float] = None,
        t_hypoxia: List[float] = None,
        mechanismID: str = "default"
    ) -> None:

        """Initializes HBS model.

        Parameters
        ----------
        parameters
            List of floats defining the parameters

        inputs
            List of floats defining the input pO2: [p_O2_hypoxia, pO2_normoxia]

        mechanismID
            a string defining the mechanism identity

        Returns
        -------
        None

        """
        self.parameters = parameters
        self.mechanismID = mechanismID

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
        self, x: list[float], topology: str
    ) -> dict:
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
        solution_hypoxia_dict
            a dict containing solutions for all model states for the given topology

        """
        # solve in normoxia
        t_normoxia = np.arange(0, 500, 1)

        solution_normoxia = odeint(
            self.topology_gradient,
            self.initial_conditions,
            t_normoxia,
            args=(
                self.parameters,
                x[1],
                topology
            ),
        )
        solution_normoxia_dict = {}
    
        for i in range(0,self.number_of_states):
            solution_normoxia_dict[self.state_labels[i]] = solution_normoxia[:,i]
        
        # solve in hypoxia
        # t_hypoxia = [0, 24, 48, 72, 96, 120]
        initial_conditions_hypoxia = []
    
        for label in self.state_labels:
            initial_conditions_hypoxia.append(solution_normoxia_dict[label][-1])
            
        solution_hypoxia_dict = {}
        for O2_val in x:
            solution_hypoxia_dict[O2_val] = {}
            solution_hypoxia = odeint(
                self.topology_gradient,
                initial_conditions_hypoxia,
                self.t_hypoxia,
                args=(
                    self.parameters,
                    O2_val,
                    topology
                ),
            )
            for i in range(0,self.number_of_states):
                solution_hypoxia_dict[O2_val][self.state_labels[i]] = solution_hypoxia[:,i]
        
        return solution_hypoxia_dict

    @staticmethod
    def topology_gradient_D(
        y: np.ndarray, 
        t: np.ndarray, 
        parameters: list[float], 
        input: float, 
        topology: str
    ) -> np.ndarray:
        """Defines the gradient for each HBS topology model.

        Parameters
        ----------
        y
            an array defining the initial conditions (necessary parameter to use ODEint to solve gradient)

        t
            an array defining the time (necessary parameter to use ODEint to solve gradient)

        parameters
            a list of floats defining the parameters

        input
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

        pO2 = input

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
            
        #parameters that will be held constant:
        k_txn = 1.0 #U
        k_dR = 2.7 #1/h
        k_tln = 1 #U
        k_dP = 0.35 #1/h
        k_dRep = 0.029 #1/hr

        if pO2 == 138.0:
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
            dydt_H1a_fb = deepcopy(dydt)
            dydt_H1a_fb[6] = k_txn + k_txnBH*(y[7] + y[10]) - k_dR*y[6] - k_dH1R*y[5]*y[6]
            return dydt_H1a_fb
        elif topology == "H2a_fb":
            # y[8] = HIF2a mRNA, CHANGE dydt[8] for H2a fb
            dydt_H2a_fb = deepcopy(dydt)
            dydt_H2a_fb[8] = k_txn + k_txnBH*(y[7] + y[10]) - k_dR*y[8]
            return dydt_H2a_fb

    def solve_experiment(
        self, 
        x: list[float],
    ) -> Tuple[dict[str, dict[int, dict[str, np.ndarray]]], float]:
        """Solve all HBS topology models in normoxia and hypoxia.

        Parameters
        ----------
        x
        a list of floats containing the values of the independent variable

        Returns
        -------
        all_hypoxia_dict
        a dict of dicts with solutions for all model states for each topology

        """
        topologies = ["simple", "H1a_fb", "H2a_fb"]
        self.t_hypoxia = [0.0, 24.0, 48.0, 72.0, 96.0, 120.0]

        all_topology_hypoxia_dict = {}

        for topology in topologies:
            solution_hypoxia_dict = self.solve_single(
                x,
                topology
            )
            all_topology_hypoxia_dict[topology] = solution_hypoxia_dict

            normalization_value = np.mean(all_topology_hypoxia_dict["simple"][6.6]['DSRE2P'][:5])

        return all_topology_hypoxia_dict, normalization_value

    def solve_experiment_for_plot(
            self,
            x: List[float],
    ) -> dict[str, dict[int, dict[str, np.ndarray]]]:
        
        """Solve all HBS topology models in normoxia and hypoxia for t values
        to use in plotting. First solve with original t values for simple HBS
        to calculate normalization value.
           
        Parameters
        ----------
        x
        a list of floats containing the values of the independent variable

        Returns
        -------
        all_hypoxia_dict
        a dict of dicts with solutions for all model states for each topology
        """
        self.t_hypoxia = [0.0, 24.0, 48.0, 72.0, 96.0, 120.0]
        solution_hypoxia_dict = self.solve_single(
            x,
            "simple"
        ) #simulate simple HBS with original time points to calculate normalization value

        normalization_value = np.mean(solution_hypoxia_dict[6.6]['DSRE2P'][:5])

        self.t_hypoxia = np.linspace(0,120,31)
        topologies = ["simple", "H1a_fb", "H2a_fb"]

        all_topology_hypoxia_dict = {}

        for topology in topologies:
            solution_hypoxia_dict = self.solve_single(
                x,
                topology
            )
            all_topology_hypoxia_dict[topology] = solution_hypoxia_dict

        solutions_DsRE2P_simple = np.append(
            all_topology_hypoxia_dict["simple"][6.6]["DSRE2P"][:26],
            all_topology_hypoxia_dict["simple"][138.0]["DSRE2P"][0]
        )
        solutions_DsRE2P_H1a_fb = np.append(
            all_topology_hypoxia_dict["H1a_fb"][6.6]["DSRE2P"],
            all_topology_hypoxia_dict["H1a_fb"][138.0]["DSRE2P"][0]
        )
        solutions_DsRE2P_H2a_fb = np.append(
            all_topology_hypoxia_dict["H2a_fb"][6.6]["DSRE2P"],
            all_topology_hypoxia_dict["H2a_fb"][138.0]["DSRE2P"][0]
        )

        normalized_DsRE2P_simple = self.normalize_data(
            solutions_DsRE2P_simple,
            normalization_value
        )
        normalized_DsRE2P_H1a_fb = self.normalize_data(
            solutions_DsRE2P_H1a_fb,
            normalization_value
        )
        normalized_DsRE2P_H2a_fb = self.normalize_data(
            solutions_DsRE2P_H2a_fb,
            normalization_value
        )

        all_topology_DsRE2P = [normalized_DsRE2P_simple, normalized_DsRE2P_H1a_fb, normalized_DsRE2P_H2a_fb]

        return all_topology_hypoxia_dict, all_topology_DsRE2P
    
    @staticmethod
    def normalize_data(solutions_raw: np.ndarray, normalization_value: float) -> np.ndarray:
        """Normalizes data by mean simple HBS value

        Parameters
        ----------
        solutions_raw
            a 1D array of floats defining the solutions before normalization

        normalization_value
            a float defining the mean of the simple HBS hypoxia simulation
            (to be used for normalization)

        Returns
        -------
        solutions_norm
            a 1D array of floats defining the dependent variable for the given
            dataset (after normalization)

        """

        solutions_norm = solutions_raw/normalization_value
        
        return solutions_norm

    @staticmethod
    def plot_training_data(
        solutions_norm: List[np.ndarray],
        exp_data: List[float],
        exp_error: List[float],
        filename: str,
        run_type: str,
        context: str,
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
            plot_color1 = "black"
            plot_color2 = "gray"
            marker_type = "o"

        elif run_type == "PEM evaluation":
            plot_color1 = "dimgrey"
            plot_color2 = "lightgray"

            marker_type = "^"

        plot_settings = plot_color1, plot_color2, marker_type
        plot_training_data_2d(
            solutions_norm,
            exp_data,
            exp_error, 
            filename, 
            plot_settings, 
            context
        )

  
