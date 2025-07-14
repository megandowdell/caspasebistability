# nullclines.py
# --------------------------------------------------
# Handles plotting and solving nullcline intersections
# for the 2D reduced system (dx2, dx4)
# --------------------------------------------------

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.lines import Line2D
import warnings

from analysis import analyze, classify_stability
from models import dx2_red, dx4_red, J2D_func, params, x2, x4


def find_and_plot_nullclines(grid_size=100):
    """
    Finds intersections of dx2 = 0 and dx4 = 0 nullclines,
    classifies their stability, and plots the results.
    """
    f2 = sp.lambdify((x2, x4), dx2_red.subs(params), "numpy")
    f4 = sp.lambdify((x2, x4), dx4_red.subs(params), "numpy")

    x2_vals = np.linspace(1e-2, 20000, grid_size)
    x4_vals = np.linspace(1e-2, 20000, grid_size)
    X2, X4 = np.meshgrid(x2_vals, x4_vals)
    Z2, Z4 = f2(X2, X4), f4(X2, X4)

    plt.figure(figsize=(10, 7))
    plt.contour(X2, X4, Z2, levels=[0], colors="blue", linestyles="--")
    plt.contour(X2, X4, Z4, levels=[0], colors="red")

    scan_points = [(i, j) for i in np.linspace(0.1, 20000, 100)
                          for j in np.linspace(0.1, 20000, 100)]

    found_ss = []
    for guess in scan_points:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                root = fsolve(lambda v: [f2(*v), f4(*v)], guess)
            except:
                continue

        if not all(0 <= r <= 20000 for r in root):
            continue
        if any(np.linalg.norm(root - s) < 1 for s in found_ss):
            continue

        dx2_val, dx4_val = f2(*root), f4(*root)
        if abs(dx2_val) > 1e-6 or abs(dx4_val) > 1e-6:
            continue

        stab = classify_stability(np.linalg.eigvals(J2D_func(*root)))
        color = {"Stable": "green", "Unstable (Source)": "red", "Saddle (Mixed)": "orange"}[stab]
        plt.plot(*root, "o", color=color)
        label = f"{stab}\n({root[0]:.1f}, {root[1]:.1f})"
        plt.text(root[0] + 10, root[1], label, fontsize=7)

        analyze(root, verbose=False, show_plot=False)
        found_ss.append(root)

    plt.xlim(0, 20000)
    plt.ylim(0, 20000)

    legend = [
        Line2D([0], [0], color='blue', linestyle='--', label=r'$\dot{x}_2 = 0$'),
        Line2D([0], [0], color='red', label=r'$\dot{x}_4 = 0$'),
        Line2D([0], [0], marker='o', color='w', label='Stable SS', markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Unstable SS', markerfacecolor='red', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Saddle SS', markerfacecolor='orange', markersize=8),
    ]
    plt.legend(handles=legend, loc="upper right")
    plt.xlabel("x2 (Active Caspase-8)")
    plt.ylabel("x4 (Active Caspase-3)")
    plt.title("Nullclines and Steady States (x₂, x₄)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
