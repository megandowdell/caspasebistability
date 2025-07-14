# main.py
# Entry point for Apoptosis Bistability Analysis

from models import define_model, reduce_model
from analysis import analyze_steady_state
from sweeping import parameter_scan, plot_scan_results

import numpy as np

if __name__ == "__main__":
    # Setup model equations and parameter dictionary
    model = define_model()
    dx2_red, dx4_red, J2D_func, J8D_func = reduce_model(model)

    # Example: Analyze a known steady state (x2, x4)
    steady_state = (211.2, 511.5)
    analyze_steady_state(steady_state, model, J2D_func, J8D_func)

    # Run a parameter scan for k1 (uncomment to use)
    # values = np.logspace(-7, 5, 100)
    # df, bistables = parameter_scan('k1', values, model, dx2_red, dx4_red, J2D_func, J8D_func)
    # plot_scan_results(df, 'k1')
