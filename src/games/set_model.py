#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 14:57:05 2022

@author: kate
"""

from config import Settings
from models import synTF, synTF_chem


def Set_Model():
    """
    Defines the model class to use depending on the modelID defined in Settings

    Parameters
    ----------
    None

    Returns
    -------
    model
        object defining the model
    """
    if Settings.modelID == "synTF_chem":
        given_model = synTF_chem(parameters=Settings.parameters)

    elif Settings.modelID == "synTF":
        given_model = synTF(parameters=Settings.parameters)

    return given_model


model = Set_Model()
