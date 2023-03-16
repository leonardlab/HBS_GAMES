#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:36:16 2022

@author: kate
"""
from typing import Tuple, List
import pandas as pd


def define_experimental_data(settings: dict) -> Tuple[List[float], List[float], List[float]]:
   """
   Imports experimental data

   Parameters
   ---------
   settings
      a dictionary of run settings

   Returns
   -------
   x
      a list of floats defining the independent variable for the given dataset

   exp_data
      a list of floats defining the normalized dependent variable for the given
      dataset

   exp_error
      a list of floats defining the normalized measurement error for
      the dependent variable for the given dataset
   """

   path = settings["context"] + "config/"
   filename = path + "training_data_" + settings["dataID"] + ".csv"
   df_exp = pd.read_csv(filename)
   # print(df_exp)
   percent_O2 = list(df_exp["%O2"])
   exp_data_all = list(df_exp["y"])
   exp_error_all = list(df_exp["y_err"])

   if settings["dataID"] == "hypoxia_only":
      exp_simple_HBS = exp_data_all[:5] + [exp_data_all[6]]
      exp_H1a_fb_HBS = exp_data_all[9:15] + [exp_data_all[15]]
      exp_H2a_fb_HBS = exp_data_all[18:24] + [exp_data_all[24]]
      
      
      error_simple_HBS = exp_error_all[:5] + [exp_error_all[6]]
      error_H1a_fb_HBS = exp_error_all[9:15] + [exp_error_all[15]]
      error_H2a_fb_HBS = exp_error_all[18:24] + [exp_error_all[24]]

   exp_data = exp_simple_HBS + exp_H1a_fb_HBS + exp_H2a_fb_HBS
   exp_error = error_simple_HBS + error_H1a_fb_HBS + error_H2a_fb_HBS

   input_pO2 = []
   for percent in percent_O2:
      if percent == 1:
         pO2 = 7.6
         input_pO2.append(pO2)
      elif percent == 21:
         pO2 = 138.0
         input_pO2.append(pO2)

   return input_pO2, exp_data, exp_error


# settings = {
#    "context": "/Users/kdreyer/Desktop/Github/HBS_GAMES2/src/games/",
#    "dataID": "hypoxia_only",
# }

# input_pO2, exp_data, exp_error = define_experimental_data(
#    settings
# )
