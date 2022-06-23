#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 14:57:05 2022

@author: kate
"""
from games.config.settings import settings
from games.models.synTF import synTF
from games.models.synTF_chem import synTF_chem


def set_model() -> any:
    """
    Defines the model class to use depending on the systemID defined in Settings

    Parameters
    ----------
    None

    Returns
    -------
    model
        object defining the model

    """

    if settings["systemID"] == "synTF_chem":
        given_model = synTF_chem(parameters=settings["parameters"])

    elif settings["systemID"] == "synTF":
        given_model = synTF(parameters=settings["parameters"])

    return given_model


model = set_model()
