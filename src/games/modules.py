#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:41:51 2022

@author: kate
"""
import os
from config import Settings, ExperimentalData, saveConditions, createFolder
from analysis import plot_x_y
from set_model import model
from global_search import generate_parameter_sets, solve_global_search
from optimization import Optimization
from test_single import solve_single_parameter_set


class Modules:
    """Class with methods to run each module from the GAMES workflow"""

    @staticmethod
    def test_single_parameter_set():
        """Solves model for a single parameter set using dataID defined in Settings.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        sub_folder_name = "TEST SINGLE PARAMETER SET"
        path = createFolder(sub_folder_name)
        os.chdir(path)
        saveConditions()
        model.parameters = Settings.parameters
        solutions_norm, chi_sq, r_sq = solve_single_parameter_set()
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
            "black",
        )
        return solutions_norm, chi_sq, r_sq

    @staticmethod
    def estimate_parameters():
        """Runs parameter estimation method (multi-start optimization)

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        sub_folder_name = "MODULE 2 - FIT TO EXPERIMENTAL DATA"
        path = createFolder(sub_folder_name)
        os.chdir(path)
        saveConditions()

        print("Starting global search...")
        df_parameters = generate_parameter_sets(Settings.parameter_estimation_problem_definition)
        df_global_search_results = solve_global_search(df_parameters)
        print("Global search complete.")

        print("Starting optimization...")
        Optimization.optimize_all(df_global_search_results)

    @staticmethod
    def generate_parameter_estimation_method_evaluation_data():
        """xxx"""
        pass

    @staticmethod
    def evaluate_parameter_estimation_method():
        """xxx"""
        pass

    @staticmethod
    def calculate_profile_likelihood():
        """xxx"""
        pass
