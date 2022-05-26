#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 09:11:06 2022

@author: kate
"""
import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from config import Settings, ExperimentalData, Context
from plots import Plots

plt.style.use("./paper.mplstyle.py")

class Modules:
    def test_single_parameter_set():
        if Settings.modelID == 'synTF_chem':
            if Settings.dataID == 'ligand dose response':
                x = ExperimentalData.x
                solutions = model.solve_ligand_sweep(x)
                solutions_norm = [i/max(solutions) for i in solutions]
                Plots.plot_x_y(x, solutions_norm, ExperimentalData.exp_data, ExperimentalData.exp_error, 'Ligand (nM)', 'Rep. protein (au)', 'ligand sweep', y_scale = 'log')
       
        elif Settings.modelID == 'synTF':
            if Settings.dataID == 'synTF dose response':
                x = ExperimentalData.x
                solutions = model.solve_synTF_sweep(x)
                solutions_norm = [i/max(solutions) for i in solutions]
                Plots.plot_x_y(x, solutions_norm, ExperimentalData.exp_data, ExperimentalData.exp_error, 'synTF', 'Rep. protein (au)', 'synTF sweep')
       
                
    def estimate_parameters():
        pass
    def evaluate_parameter_estimation_method():
        pass
    def calculate_profile_likelihood():
        pass
        

class GeneralModel:
    """
    General information for any ODE model

    """
    
    def set_general_parameters(self):
        """Defines general parameters that may be used in any model.
  
        Parameters
        ----------
        None 
        
        Returns
        -------
        None
        
        """
        k_txn = 1
        k_trans = 1
        kdeg_rna = 2.7
        kdeg_protein = 0.35
        kdeg_reporter = 0.029
        k_deg_ligand = 0.01
        self.general_parameters = np.array(
            [k_txn, k_trans, kdeg_rna, kdeg_protein, kdeg_reporter, k_deg_ligand]
        )
     

class synTF_chem(GeneralModel, Modules):
    """
    Representation of synTF_Chem model - synTFChem

    """

    def __init__(self, free_parameters = [1, 1, 1, 1, 1, 1], inputs = [50, 50], input_ligand = 1000):
        """Initializes synTF_Chem model.
        
        Parameters
        ----------
        free_parameters
            List of floats defining the free parameters
        
        inputs
            List of floats defining the inputs 
        
        input_ligand
            Float defining the input ligand concentration
            
        Returns
        -------
        None 
        
        """
        self.number_of_states = 8
        
        self.free_parameters = np.array(free_parameters)
        self.inputs = np.array(inputs)
        self.input_ligand = input_ligand
        
        GeneralModel.set_general_parameters(self)
        
        y_init = np.zeros(self.number_of_states)
        self.initial_conditions = y_init

    def solve_single(self):
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
        end_time = 18
        tspace_before_ligand_addition = np.linspace(0, end_time, timesteps)
        self.t = tspace_before_ligand_addition
        solution_before_ligand_addition = odeint(
            self.gradient,
            self.initial_conditions,
            self.t,
            args=(
                self.free_parameters,
                self.inputs,
                self.general_parameters,
            ),
        )

        # solve after ligand addition
        end_time = 24
        tspace_after_ligand_addition = np.linspace(0, end_time, timesteps)
        self.t = tspace_after_ligand_addition
        initial_conditions_after_ligand_addition = np.array(solution_before_ligand_addition[-1, :])
        initial_conditions_after_ligand_addition[4] = self.input_ligand
        self.initial_conditions_after_ligand_addition = initial_conditions_after_ligand_addition
        solution_after_ligand_addition = odeint(
            self.gradient,
            self.initial_conditions_after_ligand_addition,
            self.t,
            args=(
                self.free_parameters,
                self.inputs,
                self.general_parameters,
            ),
        )
        self.solution = solution_after_ligand_addition

        return self.solution, self.t

    def gradient(self, y, t, free_parameters, inputs, general_parameters):
        """Defines the gradient for synTF_Chem model.
        
        Parameters
        ----------
        free_parameters
            List of floats defining the free parameters
        
        inputs
            List of floats defining the inputs 
            
        general_parameters
            List of floats defining the general parameters that may be used
                k_txn, k_trans, kdeg_rna, kdeg_protein, kdeg_reporter, k_deg_ligand
        
            
        Returns
        -------
        dydt
            An list of floats corresponding to the gradient of each model state at time t
        
        """
        y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8 = y
        e, b, k_bind, m, km, n = free_parameters
        dose_a, dose_b = inputs
        k_txn, k_trans, kdeg_rna, kdeg_protein, kdeg_reporter, k_deg_ligand = general_parameters

        f_num = b + m * (y_6 / km) ** n
        f_denom = 1 + (y_6 / km) ** n + (y_2 / km) ** n
        f = f_num / f_denom

        if math.isnan(f):
            f = 0

        u_1 = k_txn * dose_a - kdeg_rna * y_1  # y1 A mRNA
        u_2 = k_trans * y_1 - kdeg_protein * y_2 - k_bind * y_2 * y_4 * y_5  # y2 A protein
        u_3 = k_txn * dose_b - kdeg_rna * y_3  # y3 B mRNA
        u_4 = k_trans * y_3 - kdeg_protein * y_4 - k_bind * y_2 * y_4 * y_5  # y4 B protein
        u_5 = -k_bind * y_2 * y_4 * y_5 - y_5 * k_deg_ligand  # y5 Ligand
        u_6 = k_bind * y_2 * y_4 * y_5 - kdeg_protein * y_6  # y6 Activator
        u_7 = k_txn * f - kdeg_rna * y_7  # y7 Reporter mRNA
        u_8 = k_trans * y_7 - kdeg_reporter * y_8  # y8 Reporter protein
        dydt = np.array([u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8])

        return dydt
    
    def solve_ligand_sweep(self, x_ligand):
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
        for ligand in x_ligand:
            self.input_ligand = ligand
            sol, t = model.solve_single()
            solutions.append(sol[-1, -1])
            
        return solutions    


class synTF(GeneralModel, Modules):
    """
    Representation of synTF model - synTF only

    """

    def __init__(self, free_parameters = [1], inputs = [50]):
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
     
        self.number_of_states = 4
        self.free_parameters = np.array(free_parameters)
        self.inputs = np.array(inputs)
        GeneralModel.set_general_parameters(self)
        y_init = np.zeros(self.number_of_states)
        self.initial_conditions = y_init
  
    def solve_single(self):
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
        self.t = tspace
        self.solution = odeint(
            self.gradient,
            self.initial_conditions,
            self.t,
            args=(
                self.free_parameters,
                self.inputs,
                self.general_parameters,
            ),
        )

        return self.solution, self.t

    def gradient(self, y, t, free_parameters, inputs, general_parameters):
        """Defines the gradient for synTF model.
        
        Parameters
        ----------
        free_parameters
            List of floats defining the free parameters
        
        inputs
            List of floats defining the inputs 
            
        general_parameters
            List of floats defining the general parameters that may be used
                k_txn, k_trans, kdeg_rna, kdeg_protein, kdeg_reporter, k_deg_ligand
        
            
        Returns
        -------
        dydt
            An list of floats corresponding to the gradient of each model state at time t
        
        """
        y_1, y_2, y_3, y_4 = y
        [g] = free_parameters
        [dose_a] = inputs

        k_txn, k_trans, kdeg_rna, kdeg_protein, kdeg_reporter, k_deg_ligand = general_parameters
        
        u_1 = k_txn * dose_a - kdeg_rna * y_1  # y1 synTF mRNA
        u_2 = k_trans * y_1 - kdeg_protein * y_2  # y2 synTF protein
        u_3 = k_txn * g * y_2 - kdeg_rna * y_3  # y3 Reporter mRNA
        u_4 = k_trans * y_3 - kdeg_reporter * y_4  # y4 Reporter protein
        dydt = np.array([u_1, u_2, u_3, u_4])

        return dydt
    
    def solve_synTF_sweep(self, x):
        """Solve synTF model for a list of synTF values.
        
        Parameters
        ----------
        None
            
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
            sol, t = model.solve_single()
            solutions.append(sol[-1, -1])
            
        return solutions
   
def Set_Model():
    if Settings.modelID == 'synTF_chem':
        model = synTF_chem(free_parameters = Settings.free_parameters)
       
    elif Settings.modelID == 'synTF':
        model = synTF(free_parameters = Settings.free_parameters)
        
    return model

model = Set_Model()


