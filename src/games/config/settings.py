from math import log10
import json
from games.utilities.saving import make_main_directory


def define_parameter_estimation_problem(settings_import):
    """Defines the parameter estimation problem and structures as a dictionary"""
    free_parameters = []
    free_parameter_indicies = []
    for i, value in enumerate(settings_import["parameters"]):
        label = settings_import["parameter_labels"][i]
        if label in settings_import["free_parameter_labels"]:
            free_parameters.append(value)
            free_parameter_indicies.append(i)

    # Set default parameter bounds
    num_free_params = len(free_parameters)
    bounds_log = []
    for i in range(0, num_free_params):
        min_bound = log10(free_parameters[i]) - settings_import["bounds_orders_of_magnitude"]
        max_bound = log10(free_parameters[i]) + settings_import["bounds_orders_of_magnitude"]
        bounds_log.append([min_bound, max_bound])

    # Set non-default parameter bounds
    for key, values in settings_import["non_default_bounds"].items():
        for i, parameter_label in enumerate(settings_import["free_parameter_labels"]):
            if key == parameter_label:
                bounds_log[i] = values

    parameter_estimation_problem_definition = {
        "num_vars": num_free_params,  # set free parameters and bounds
        "names": settings_import["free_parameter_labels"],
        "bounds": bounds_log,  # bounds are in log scale
    }

    settings_import["free_parameters"] = free_parameters
    settings_import["free_parameter_indicies"] = free_parameter_indicies

    return parameter_estimation_problem_definition, settings_import


file = open("./src/games/config/config.json", encoding="utf-8")
settings_import = json.load(file)
parameter_estimation_problem_definition, settings = define_parameter_estimation_problem(
    settings_import
)
folder_path = make_main_directory(settings)
