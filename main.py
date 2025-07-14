# main.py
# Entry point for Apoptosis Bistability Analysis

import argparse
import numpy as np

from models import define_model, reduce_model
from analysis import analyze_steady_state
from sweeping import parameter_scan, plot_scan_results
from nullclines import run_nullclines
from dx4_only import run_dx4_only

def main():
    parser = argparse.ArgumentParser(description="Apoptosis Bistability Simulation")
    parser.add_argument("--mode", type=str, choices=["nullclines", "dx4", "scan", "analyze"], required=True)
    args = parser.parse_args()

    # Build model
    model = define_model()
    dx2_red, dx4_red, J2D_func, J8D_func = reduce_model(model)

    if args.mode == "nullclines":
        run_nullclines(model, dx2_red, dx4_red, J2D_func)
    elif args.mode == "dx4":
        run_dx4_only(model, dx2_red, dx4_red)
    elif args.mode == "scan":
        values = np.logspace(-7, 5, 100)
        df, bistables = parameter_scan('k1', values, model, dx2_red, dx4_red, J2D_func, J8D_func)
        plot_scan_results(df, 'k1')
    elif args.mode == "analyze":
        test_ss = (211.2, 511.5)
        analyze_steady_state(test_ss, model, J2D_func, J8D_func)

if __name__ == "__main__":
    main()
