# Introduction
This repository contains the MATLAB code used to generate the numerical results in my paper titled "Power-Constrained Transmission Design for 
Over-the-Air Collaborative Estimation". Plaintext versions of the .m files can be found in the .dat Files folder.

# Generating Your Own Results
- In `power_control.m`, change the experiments inside `experiment_list` to run the desired experiments corresponding to the paper.
- All other parameters are tailored to match the exact implementation used in the paper.
- The individual experiment plots will be generated and saved to the path specified in "folder_path".
- Run the corresponding "plot_..." script to from the `scripts` folder to plot the figures for a given experiment.
- The "plot_..." scripts will specify what result files are required to generate the figures.
- Additionally, `run_plots.m` will run all plot scripts.

If you encounter any issues with the script, please feel free to reach out to me at `vcvhuynh (at) ucdavis (dot) edu`.

# Known Errata
A list of known errata will be compiled following the initial submission of the manuscript for review and their effects on the experimental results 
will be noted below.

## 8/4/2026
- Various functions were being called from the separate directory called `util_functions`. We have updated the functions inside the `functions` directory 
  with the function definitions from `util_functions`. We note that there were primarily semantic differences between the function definitions, which 
  have no effect on the experimental results.