import numpy as np
import sympy as sp
from scipy.optimize import brentq
import matplotlib.pyplot as plt

from models import dx2_red, dx4_red, params, x2, x4
from analysis import analyze, classify_stability
from utils import ensure_dir

def run_dx4_only_plot():
    print("\nRunning 1D Reduction via dx4(x4) using dx2 = 0 constraint...")

    x2_solutions = sp.solve(sp.Eq(dx2_red, 0), x2)
    real_solutions = [sol for sol in x2_solutions if not sol.has(sp.I)]

    if not real_solutions:
        raise ValueError("❌ No real solutions to dx2 = 0 found.")

    x4_vals = np.linspace(1, 10000, 1200)
    roots_found = []
    all_y = []

    plt.figure(figsize=(10, 6))

    for i, sol in enumerate(real_solutions):
        dx4_1d = dx4_red.subs(x2, sol)
        dx4_func = sp.lambdify(x4, dx4_1d.subs(params), "numpy")

        try:
            dx4_y = dx4_func(x4_vals)
            all_y.append(dx4_y)
        except Exception:
            continue

        plt.plot(x4_vals, dx4_y, label=f"Branch {i+1}")

        for j in range(len(x4_vals) - 1):
            if dx4_y[j] * dx4_y[j + 1] < 0:
                try:
                    root = brentq(dx4_func, x4_vals[j], x4_vals[j + 1])
                    if any(abs(root - r) < 10 for r in roots_found):
                        continue

                    slope = (dx4_func(root + 1e-2) - dx4_func(root - 1e-2)) / 2e-2
                    stab = "Stable" if slope < 0 else "Unstable"
                    color = "green" if stab == "Stable" else "red"

                    plt.plot(root, 0, "o", color=color)
                    plt.text(root + 20, 0.05, f"{stab}\n({root:.1f})", fontsize=7)

                    x2_val = real_solutions[0].subs(x4, root)
                    x2_val = float(x2_val.evalf(subs=params))

                    print(f"\nAnalyzing SS: x2 = {x2_val:.2f}, x4 = {root:.2f}")
                    analyze([x2_val, root], verbose=True, show_plot=False)
                    roots_found.append(root)
                except:
                    continue

    plt.axhline(0, color="gray", linestyle="--")
    flat_y = np.hstack(all_y) if all_y else [0]
    plt.xlim(0, 6000)
    plt.ylim(min(flat_y) - 10, max(flat_y) + 10)
    plt.title("Reduced 1D Phase Plane: $\\dot{x}_4(x_4)$ with $\\dot{x}_2 = 0$")
    plt.xlabel("x4 (Active Caspase-3)")
    plt.ylabel("$\\dot{x}_4$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
