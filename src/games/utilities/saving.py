#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:47:26 2022

@author: kate
"""

import os
from config import Settings, Context

def create_folder(sub_folder_name) -> str:
    """Create a new folder.

    Parameters
    ----------
    sub_folder_name
        string defining the name of the folder to make

    Returns
    -------
    path
        path leading to new folder
    """

    path = Context.folder_path + "/" + sub_folder_name
    os.makedirs(path)

    return path


def save_conditions() -> None:
    """Save conditions for the given run

    Parameters
    ----------
    None

    Returns
    -------
    None

    """

    with open("CONDITIONS" + ".txt", "w", encoding="utf-8") as file:
        file.write("dataID: " + str(Settings.dataID) + "\n")
        file.write("\n")
        file.write("modelID: " + str(Settings.modelID) + "\n")
        file.write("\n")
        file.write(
            "parameter_estimation_problem_definition:"
            + str(Settings.parameter_estimation_problem_definition)
            + "\n"
        )
        file.write("\n")