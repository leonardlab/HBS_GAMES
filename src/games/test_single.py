#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 15:25:47 2022

@author: kate
"""
from set_model import model
from config import ExperimentalData, Settings
from analysis import CalculateMetrics


class TestSolveSingle():
    def solve_single_parameter_set():
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
        
        if Settings.modelID == 'synTF_chem':
            if Settings.dataID == 'ligand dose response':
                solutions = model.solve_ligand_sweep(ExperimentalData.x)
               
        elif Settings.modelID == 'synTF':
            if Settings.dataID == 'synTF dose response':
                solutions = model.solve_synTF_sweep(ExperimentalData.x)
                
        solutions_norm = [i/max(solutions) for i in solutions]
        chi_sq = CalculateMetrics.calc_chi_sq(ExperimentalData.exp_data, solutions_norm, ExperimentalData.exp_error)
        r_sq = CalculateMetrics.calc_r_sq(ExperimentalData.exp_data, solutions_norm)
        
        return solutions_norm, chi_sq, r_sq 