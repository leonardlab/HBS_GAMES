#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 14:57:05 2022

@author: kate
"""
import json
from games.models.synTF_chem import synTF_chem
from games.models.HBS import HBS_model


def set_model():
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

    if settings["modelID"] == "synTF_chem":
        given_model = synTF_chem(
            parameters=settings["parameters"], mechanismID=settings["mechanismID"]
        )

    elif settings["modelID"] == "HBS":
        given_model = HBS_model(
            parameters=settings["parameters"], mechanismID=settings["mechanismID"]

        )

    return given_model


file = open("/Users/kdreyer/Desktop/Github/HBS_GAMES2/src/games/config/config_HBS_D2.json", encoding="utf-8")
settings = json.load(file)
model = set_model()
