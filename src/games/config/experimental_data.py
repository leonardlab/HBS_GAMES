#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:36:16 2022

@author: kate
"""
from typing import Tuple
import pandas as pd
from games.config.settings import settings

def define_experimental_data() -> Tuple[list, list, list]:
	""""
	Imports, normalizes, and defines experimental data

	Parameters
	----------
	None

	Returns
	-------
	x
		a list of floats containing the values of the independent variable

	exp_data
		a list of floats containing the values of the dependent variable

	exp_error
		a list of floats containing the values of the measurement error
		for the dependent variable
	"""

	x, exp_data_raw, exp_error_raw = import_data()
	exp_data, exp_error = normalize_data(exp_data_raw, exp_error_raw)

	return x, exp_data, exp_error

def import_data() -> Tuple[list, list, list]:
	"""Imports experimental data

	Parameters
	----------
	None

	Returns
	-------
	x
	a list of floats defining the independent variable for the given dataset

	exp_data_raw
	a list of floats defining the dependent variable for the given
	dataset (before normalization)

	exp_error_raw
	 a list of floats defining the measurement error for
	 the dependent variable for the given dataset (before normalization)
	"""
	path = settings["context"] + "config/"
	filename = path + "training_data_" + settings["dataID"] + ".csv"
	df_exp = pd.read_csv(filename)
	x = list(df_exp["x"])
	exp_data_raw = list(df_exp["y"])
	exp_error_raw = list(df_exp["y_err"])

	return x, exp_data_raw, exp_error_raw

def normalize_data(exp_data_raw: list, exp_error_raw: list) -> Tuple[list, list]:
	"""Normalizes experimental data by maximum value

	Parameters
	----------
	exp_data_raw
	a list of floats defining the dependent variable for the given
	dataset (before normalization)

	exp_error_raw
	 a list of floats defining the measurement error for
	 the dependent variable for the given dataset (before normalization)

	Returns
	-------
	exp_data
	a list of floats defining the dependent variable for the given
	dataset (after normalization)

	exp_error
	 a list of floats defining the measurement error for
	 the dependent variable for the given dataset (after normalization)
	"""

	exp_data = []
	exp_error = []
	max_val = max(exp_data_raw)
	for i, val in enumerate(exp_data_raw):
		exp_data.append(val / max_val)
		exp_error.append(exp_error_raw[i] / max_val)
	return exp_data, exp_error
