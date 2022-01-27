# GAMES

This code is associated with the pre-print manuscript "GAMES: A dynamic model development workflow for rigorous characterization of synthetic genetic systems".

Detailed descriptions of the files and functions in this repository are included in the Supplementary Information (Supplementary Notes 2-3) of the manuscript. Simulation outputs associated with the manuscript are included as Supplementary Data 1.
Test.py and Run.py are the executable files. Settings.py can be used to change the settings for different simulation runs.

After cloning the repository, the user must create a folder called "Results" - the code will automatically save all simulation outputs in this folder. The absolute file path for this folder must be updated in the file Saving.py. Absolute paths may also need to be provided when importing REFERENCE TRAINING DATA.py and paper.mplstyle.py.

WINDOWS USERS: The parallelization component of this code is written using Python’s multiprocessing package, which has a different implementation for Mac and Windows operating systems. This code was run and tested with a Mac OS and is not set up to support Windows users. If you are a Windows user, we recommend setting parallelization = ‘no’ on line 79 in Run.py to use a version of the code that does not use parallelization. This option can be used for modules 1 and 2, but not for module 3 due to computational efficiency limitations.

RELEASE NOTE v1.0.1: The previous release of this code included an incorrect standard error value to define the error distribution used to randomly add noise to data points, leading to a slightly smaller error distribution. This new release uses the appropriate value, which will slightly impact some figures relating to generation of PEM evaluation data and calculation of PPL thresholds. To reproduce the figures exactly as in the manuscript, please use the initial release of the code (v1.0.0).

PACKAGE VERSIONS: The GAMES code uses the following versions of each Python package and was not tested with other versions: python 3.7, lmfit 0.9.14, matplotlib 3.1.3, numpy1.18.1, salib 1.3.8, pandas 1.0.5, scipy 1.4


