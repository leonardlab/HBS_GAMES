# GAMES

This code is associated with the pre-print manuscript "GAMES: A dynamic model development workflow for rigorous characterization of synthetic genetic systems".

Detailed descriptions of the files and functions in this repository are included in the Supplementary Information (Supplementary Notes 2-3) of the manuscript. Simulation outputs associated with the manuscript are included as Supplementary Data 1.  

Test.py and Run.py are the executable files. Settings.py can be used to change the settings for different simulation runs.

After cloning the repository, the user must create a folder called "Results" - the code will automatically save all simulation outputs in this folder.

RELEASE NOTE: The previous release of this code included an incorrect standard error value to define the error distribution used to randomly add noise to data points, leading to a slightly smaller error distribution. This new release uses the appropriate value, which will slightly impact some figures relating to generation of PEM evaluation data and calculation of PPL thresholds. To reproduce the figures exactly as in the manuscript, please use the previous release of the code.

