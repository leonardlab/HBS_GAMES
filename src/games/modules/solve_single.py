#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 15:25:47 2022

@author: kate
"""
import os
from typing import Tuple
import numpy as np
from models.set_model import model
from config import ExperimentalData, Settings
from utilities.saving import save_conditions, create_folder
from analysis.metrics import calc_chi_sq, calc_r_sq
from analysis.plots import plot_x_y

class Solve_single:
    
    def __init__(self):
        pass
    
    @staticmethod
    def solve_single_parameter_set() -> Tuple[list, float, float]:
        """
        Solves model for a single parameter set
    
        Parameters
        ----------
        None
    
        Returns
        -------
        solutions_norm
            a list of floats containing the normalized simulation values corresponding to the dataID defined in Settings
    
        chi_sq
            a float defining the value of the cost function
    
        r_sq
            a float defining the value of the correlation coefficient (r_sq)
        """
    
        if Settings.modelID == "synTF_chem":
            if Settings.dataID == "ligand dose response":
                solutions = model.solve_ligand_sweep(ExperimentalData.x)
    
        elif Settings.modelID == "synTF":
            if Settings.dataID == "synTF dose response":
                solutions = model.solve_synTF_sweep(ExperimentalData.x)
    
        solutions_norm = [i / max(solutions) for i in solutions]
        chi_sq = calc_chi_sq(ExperimentalData.exp_data, solutions_norm, ExperimentalData.exp_error)
        r_sq = calc_r_sq(ExperimentalData.exp_data, solutions_norm)
    
        return solutions_norm, chi_sq, r_sq
    
    @staticmethod
    def plot_single_parameter_set() -> Tuple[list, float, float]:
        """Solves model for a single parameter set using dataID defined in Settings.

        Parameters
        ----------
        None

        Returns
        -------
        solutions_norm
            a list of floats containing the normalized simulation values corresponding to the dataID defined in Settings
    
        chi_sq
            a float defining the value of the cost function
    
        r_sq
            a float defining the value of the correlation coefficient (r_sq)

        """
        sub_folder_name = "TEST SINGLE PARAMETER SET"
        path = create_folder(sub_folder_name)
        os.chdir(path)
        save_conditions()
        model.parameters = Settings.parameters
        solutions_norm, chi_sq, r_sq = Solve_single.solve_single_parameter_set()
        filename = "FIT TO TRAINING DATA"
        plot_x_y(
            [
                ExperimentalData.x,
                solutions_norm,
                ExperimentalData.exp_data,
                ExperimentalData.exp_error,
            ],
            [ExperimentalData.x_label, ExperimentalData.y_label],
            filename,
            ExperimentalData.x_scale,
        )
  
        print('Parameters')
        for i, label in enumerate(Settings.parameter_labels):
            print(label + '= ' + str(model.parameters[i]))
        print('')
        print('Metrics')
        print("R_sq = " + str(np.round(r_sq, 4)))
        print("chi_sq = " + str(np.round(chi_sq, 4)))
        print("*************************")

        return solutions_norm, chi_sq, r_sq
