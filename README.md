# GAMES code

v2.0.0 is a refactored version of the GAMES code used in Dray et al. 2022. 
This version includes a variety of Python tools for type annotation, linting, testing, and documentation, along with a new, improved, and more user-friendly code stucture that is more amenable to extention to different models, data sets, and simulation conditions. 
Information on how to install and run each Python tool is included below.
 

## Extending the code to a new model 

To add a new model, the user must only add a new file in the models directory and a new .csv with the experimental data file.
The model file should include a class with the model name and should include the same general functions at the example shown here (synTF_chem, synTF).
Note that the model synTF is not mentioned in the GAMES paper, but is included her as an example of how to integrate multiple different models in the GAMES workflow. The experimental data file should have a similar structure to the examples provided.  


## Changing run settings 

To change run settings, the user can edit the "config.json" file and change each item as needed (for example, parameter estimation method hyperparameters or free parameters). 
The user must change the "context" value to the path to the GAMES directory on their own machine.  

## Running GAMES with the command line 

To run a given module (0 = test with a single parameter set, 1 = PEM evaluation, 2 = parameter estimation, 3 = parameter profile likelihood), use the command line to run the following, where x is the module number: 

```
$ run --modules='x' 
```

Mutiple modules can be run in series. For example,  the following command will run modules 2 and 3.  

```
$ run --modules='23' 
```

##Unit tests

We provide a small number of unit tests here as a learning tool and proof-of-principle for the testing architecture. 
The user may want to use the examples shown here to write more unit tests or functional tests based on their own needs.  

##Settings 

Descriptions of all settings in config.json

  - folder_name: a string defining the name of folder to save results to  

  - modelID:  sa tring defining the model to use, should be same name as the relevant class 

  - dataID : a string defining the data to use, .csv defining the data should be named "training_data_" + dataID, name of dataID is user-defined 

  - mechanismID : a string defining the identify of the mechanism to use, if there is only one version of a given model, this variable is unnecessary 

  - context : a string defining the absolute path to GAMES/src/games in the given context (computer) where the code will be run 

  - parameters: a list of integers defining the starting values for each parameter. If a given parameter is not free in this run, the parameter fill be fixed at the value in this list 

  - parameters_reference : a list of integers defining the reference values for each parameter, only necessary for proof-of-principle demonstrations such that the parameter used to define the training data are known 

  - parameter_labels : a list of strings defining the labels for the parameters defined in the "parameters" variable  

  - free_parameter_labels": a list of strings defining the labels for the parameters that are free in this run 

  - bounds_orders_of_magnitude": an integer defining the orders of magnitude in each direction that parameters are allowed to vary, all free parameters have these bounds by default 

  - non_default_bounds: a dictionary defining parameters  that have non-default bounds – key is the parameter label and value is a list with the minimum bound as the first item and the maximum bound as the second item 

  - num_parameter_sets_global_search" an integer defining the number of parameter sets in the global search 

  - num_parameter_sets_optimization : an integer defining the number of initial guesses for optimization 

  - weight_by_error: a string ("yes" or "no") defining whether the cost function is weighted by measurement error 

  - num_pem_evaluation_datasets: an integer defining the number of pem evaluation data sets to generate 

  - parallelization : a string ("yes" or "no") defining whether the run should be parallelized 

  - num_cores : an integer defining the number of cores to parallelize the run across, not relevant if parallelization = 'no' 

  - num_noise_realizations : an integer defining the number of noise realizations to use to define the PPL threshold 

  - parameter_labels_for_ppl : a list of strings defining the parameter labels for which the PPL should be calculated 

  - default_min_step_fraction_ppl : a float defining the default fraction of the calibrated value to set the minimum step for PPL 

  - non_default_min_step_fraction_ppl : a dictionary defining non-default min step values for PPL – key is the parameter label and value is a list with the direction (-1 or 1) as the first item and fraction as the second value 

  - default_max_step_fraction_ppl : a floatdefining the default fraction of the calibrated value to set the maximum step for PPL 

  - non_default_max_step_fraction_ppl : a dictionary defining non-default max step values for PPL – key is the parameter label and value is a list with the direction (-1 or 1) as the first item and fraction as the second value 

  - default_max_number_steps_ppl: an integer defining the default maximum number of PPL steps in each direction 

  - non_default_number_steps_ppl : a dictionary defining non-default maximum number of PPL steps – key is the parameter label and value is a list with the direction (-1 or 1) as the first item and number of steps as the second value 

 

# Python project tools

This repositor uses the GitHub Actions and the following tools:

- [Poetry](https://python-poetry.org/) for packaging and dependency management
- [Tox](https://tox.readthedocs.io/en/latest/) for automated testing
- [Black](https://black.readthedocs.io/en/stable/) for code formatting
- [Pylint](https://www.pylint.org/) for linting
- [Mypy](http://mypy-lang.org/) for type checking
- [Sphinx](https://www.sphinx-doc.org/) for automated documentation

Make sure you have Poetry installed.
The other tools will be installed by Poetry.

## Getting started

1. Clone the repo.
2. Initialize the repository (if you already have a `pyproject.toml` file, you can skip this step):

```bash
$ poetry init
```

3. Activate the environment (this is all you need for day-to-day development):

```bash
$ poetry shell
```

4. Install dependencies.

```bash
$ poetry install
```

5. Run a test with the CLI.

```bash
$ run --modules='0' 
```

## General commands

The `Makefile` include three commands for working with the project.

- `make clean` will clean all the build and testing files
- `make build` will run tests, format, lint, and type check your code (you can also just run `tox`)
- `make docs` will generate documentation

## Template updates

There are a number of places in this template reposiotry that are specific to the template and will need to be updated for your specific project:

- Badge links in the `README.md`
- Section `[tool.poetry]` in `pyproject.toml`
- Project information section in `docs/conf.py`

## Repository tools

### Poetry

Poetry makes it explicit what dependencies (and what versions of those dependencies) are necessary for the project.
When new dependencies are added, Poetry performs an exhaustive dependency resolution to make sure that all the dependencies (and their versions) can work together.
This does mean the initial install can take a little while, especially if you have many dependencies, but subsequent installs will be faster once the `poetry.lock` file has been created.

To add a dependency, use:

```bash
$ poetry add <dependency>
```

You can additionally specify version constraints (e.g. `<dependency>@<version constraints>`).
Use `-D` to indicate development dependencies.
You can also add dependencies directly to the  file.


### GitHub Actions

Tests are run on each push.
For projects that should be tested on multiple Python versions, make sure to update the matrix with additional versions in `.github/workflows/build.yml`.

Documentation is automatically generated by `.github/workflows/documentation.yml` on pushes to the main branch.
The documentation files are deployed on a separate branch called `gh-pages`.
You can host the documentation using GitHub Pages (Settings > Pages) from this branch.

Linting is performed on each push.
This workflow `.github/workflows/lint.yml` lints code using Pylint (fails when score is < 7.0), checks formatting with Black (fails if files would be reformatted), and performs type checking with MyPy (fails if code has type errors).
Note that this type checking is not the same as the type checking done by Tox, which additionally checks for missing types.

### Tox

Tox aims to automate and standardize testing.
You can use tox to automatically run tests on different python versions, as well as things like linting and type checking.

Tox can be configured in `tox.ini` for additional python versions or testing environments.
Note that the type checking specified in the provided `tox.ini` is more strict than the type checking specified in `.github/workflows/lint.yml`.

You can run specific tox environments using:

```bash
$ tox -e <env>
```

### Pylint

Pylint checks for basic errors in your code, aims to enforce a coding standard, and identifies code smells.
The tool will score code out of 10, with the linting GitHub Action set to pass if the score is above 7.
Most recommendations from Pylint are good, but it is not perfect.
Make sure to be deliberate with which messages you ignore and which recommendations you follow.

Pylint can be configured in `.pylintrc` to ignore specific messages (such as `missing-module-docstring`), exclude certain variable names that Pylint considers too short, and adjust additional settings relevant for your project.

### Mypy

Mypy performs static type checking.
Adding type hinting makes it easier to find bugs and removes the need to add tests solely for type checking.

### Sphinx

Sphinx is a tool to generate documentation.
We have set it up to automatically generate documenation from [Numpy style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).
It will also pull `README.md` into the main page.

Note that the documentation workflow `.github/workflows/documentation.yml` does not import dependencies, which will break the building process.
To avoid this, make sure to list your external dependencies in `conf.py` in the `autodoc_mock_imports` variable.

### Codecov

To use Codecov, you must set up the repo on [app.codecov.io](app.codecov.io) and add the code code token (`CODECOV_TOKEN`) as a repository secret.
Make sure to also up the badge token (not that same as the secret token!) in your README.
Coverage results from `.github/workflows/build.yml` will be automatically uploaded.
