# scanning.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from analysis import analyze, classify_stability
from utils import ensure_dir


def parameter_scan(param_to_vary, values_to_test, dx2_func, dx4_func,
                   params, filename=None, plot_dir=None, master_csv_path=None):
    """
    Sweeps a parameter across specified values and analyzes steady states.
    Stores results to CSV and optionally plots a bifurcation-style diagram.
    """
    if filename is None:
        filename = f"scan_results_{str(param_to_vary)}.csv"

    results = []
    for val in values_to_test:
        local_params = dict(params)
        local_params[param_to_vary] = val

        f2 = dx2_func.subs(local_params)
        f4 = dx4_func.subs(local_params)
        f2_l = sp.lambdify((x2, x4), f2, "numpy")
        f4_l = sp.lambdify((x2, x4), f4, "numpy")

        found = []
        scan_pts = [(i, j) for i in np.linspace(0.1, 6000, 60) for j in np.linspace(0.1, 6000, 60)]

        for guess in scan_pts:
            try:
                root = fsolve(lambda v: [f2_l(*v), f4_l(*v)], guess)
            except:
                continue

            if not all(0 <= r <= 1e5 for r in root):
                continue
            if any(np.linalg.norm(root - s) < 1 for s in found):
                continue

            if abs(f2_l(*root)) > 1e-6 or abs(f4_l(*root)) > 1e-6:
                continue

            try:
                analysis = analyze(root, verbose=False, show_plot=False)
            except:
                continue

            row = {
                "param_value": val,
                "x2_ss": root[0],
                "x4_ss": root[1],
                "stab_2D": analysis["stab2d"],
                "stab_8D": analysis["stab8d"],
                "conflict": analysis["conflict"]
            }

            results.append(row)
            found.append(root)

            if master_csv_path:
                pd.DataFrame([row]).to_csv(master_csv_path, mode='a', index=False, header=not os.path.exists(master_csv_path))

    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

    if plot_dir:
        plot_bifurcation_x4_only(df, param_to_vary, plot_dir)

    return df


def plot_bifurcation_x4_only(df, param_name, plot_dir):
    """
    Plot x4 steady states vs param values colored by stability type.
    """
    plt.figure(figsize=(10, 6))
    stab_colors = {
        "Stable": "green",
        "Unstable (Source)": "red",
        "Saddle (Mixed)": "orange"
    }

    for stab in stab_colors:
        sub = df[df["stab_8D"] == stab]
        plt.scatter(sub["param_value"], sub["x4_ss"], label=stab, color=stab_colors[stab], alpha=0.8)

    plt.xscale("log")
    plt.xlabel(f"Parameter: {str(param_name)}")
    plt.ylabel("x₄ Steady State")
    plt.title("Bifurcation Diagram (x₄ vs parameter)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(plot_dir, f"{str(param_name)}_x4_only.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
