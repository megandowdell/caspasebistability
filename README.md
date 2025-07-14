# caspasebistability

This project implements and analyzes a mathematical model of bistability in apoptotic signaling. The core system is based on an 8-dimensional system of ODEs from the literature, with symbolic reductions used to explore lower-dimensional dynamics. The codebase supports steady-state analysis, nullcline visualization, 1D reductions, and parameter sweeps.

Repository Structure

  main.py: Primary script to run simulations or analysis. Choose mode using the --mode argument.

  models.py: Defines the full ODE model, parameter dictionary, and symbolic reductions.

  analysis.py: Contains the core analysis logic to compute steady states, eigenvalues, and classify stability.

  nullclines.py: Plots 2D nullclines (dx2 = 0, dx4 = 0) and identifies intersection points and their stability.

  dx4_only.py: Reduces the system to 1D using dx2 = 0 constraint and analyzes dx4(x4) structure.

  sweeping.py: Implements parameter sweep and bifurcation-style diagram generation.

  utils.py: Helper functions for output directories, formatting, and other common operations.


Requirements:

Install dependencies using pip:

pip install numpy sympy scipy matplotlib pandas


How to Run:

Use main.py with the --mode flag to run specific components:

python main.py --mode nullclines   # Plots dx2/dx4 nullclines and finds steady states
python main.py --mode dx4          # Performs 1D reduction analysis of dx4(x4)
python main.py --mode scan         # Runs parameter sweep for k1 (default)
python main.py --mode analyze      # Analyzes a specified steady state in 2D and 8D

Results will be shown as plots and optionally written to CSV/image files (if enabled).


Notes:

The parameter set is the modified version of the parameters based on Waldherr et al. and may be adjusted in models.py.

All symbolic computations are handled by SymPy and evaluated numerically using SciPy.

Plots and figures are designed for both exploratory analysis and paper inclusion.
