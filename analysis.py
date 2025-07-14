# analysis.py
import numpy as np
from sympy import lambdify
import matplotlib.pyplot as plt
from models import params, vars_full, dx2_red, dx4_red, J2D, J8D, x2, x4, sols
import os

STABILITY_TOL = 1e-4

J2D_func = lambdify((x2, x4), J2D.subs(params), "numpy")
J8D_func = lambdify(vars_full, J8D.subs(params), "numpy")

def classify_stability(eigenvalues, tol=STABILITY_TOL):
    real_parts = np.real(eigenvalues)
    num_pos = np.sum(real_parts > tol)
    num_neg = np.sum(real_parts < -tol)

    if num_pos == 0 and num_neg > 0:
        return "Stable"
    elif num_neg == 0 and num_pos > 0:
        return "Unstable (Source)"
    else:
        return "Saddle (Mixed)"

def analyze(ss, verbose=True, show_plot=True, eigplot_path=None):
    """
    Analyze stability at a given (x2, x4) steady state:
    - Back-substitute full 8D state
    - Evaluate Jacobians and eigenvalues
    - Optionally plot and/or save spectrum
    """
    x2_val, x4_val = ss
    subs_dict = {x2: x2_val, x4: x4_val}

    try:
        full_state = [
            float(sols[0].get(var, var).subs(subs_dict).subs(params))
            for var in vars_full
        ]
        if any(v < 0 for v in full_state):
            if verbose:
                print("⚠️ Skipping: Negative concentrations in full state.")
            return None
    except Exception as e:
        if verbose:
            print(f"❌ Could not evaluate full state: {e}")
        return None

    jac2d = J2D_func(x2_val, x4_val)
    eigs2d = np.linalg.eigvals(jac2d)
    stab2d = classify_stability(eigs2d)

    jac8d = J8D_func(*full_state)
    eigs8d = np.linalg.eigvals(jac8d)
    stab8d = classify_stability(eigs8d)

    if verbose:
        print(f"\nAnalyzing at SS (x₂ = {x2_val:.4f}, x₄ = {x4_val:.4f})")
        print("----------------------------------------------------")
        print("  → 2D eigenvalues:", np.round(eigs2d, 5), "→", stab2d)
        print("  → 8D eigenvalues:", np.round(eigs8d, 5), "→", stab8d)
        match_str = "✅ 2D and 8D match" if stab2d == stab8d else "❌ 2D/8D mismatch"
        print(f"  {match_str}")

    if eigplot_path:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.axvline(0, color="gray", linestyle="--")
        ax.scatter(np.real(eigs8d), np.imag(eigs8d), color="purple")
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.set_title("8D Eigenvalue Spectrum")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(eigplot_path, dpi=300)
        plt.close(fig)
    elif show_plot:
        plt.figure(figsize=(6, 4))
        plt.axvline(0, color="gray", linestyle="--")
        plt.scatter(np.real(eigs8d), np.imag(eigs8d), color="purple")
        plt.xlabel("Re(λ)")
        plt.ylabel("Im(λ)")
        plt.title("8D Eigenvalue Spectrum")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "state": (x2_val, x4_val),
        "full_state": full_state,
        "jac2d": jac2d,
        "eigvals2d": eigs2d,
        "stab2d": stab2d,
        "jac8d": jac8d,
        "eigvals8d": eigs8d,
        "stab8d": stab8d,
        "conflict": stab2d != stab8d
    }
