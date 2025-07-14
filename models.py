import sympy as sp

# Define all symbolic variables
x1, x2, x3, x4, x5, x6, x7, x8 = sp.symbols("x1 x2 x3 x4 x5 x6 x7 x8")
k1, k2, k3, k4, k5, k6, k7, k8 = sp.symbols("k1 k2 k3 k4 k5 k6 k7 k8")
k9, k10, k11, k12, k13 = sp.symbols("k9 k10 k11 k12 k13")
l3, l8, l9, l10, l11, l12 = sp.symbols("l3 l8 l9 l10 l11 l12")

# Pack variables and parameters
vars_full = [x1, x2, x3, x4, x5, x6, x7, x8]
param_symbols = [k1, k2, k3, k4, k5, k6, k7, k8,
                 k9, k10, k11, k12, k13, l3, l8, l9, l10, l11, l12]

# Define parameter dictionary
params = {'k1': 1.42e-5, 'k2': 1e-5, 'k3': 5e-4, 'k4': 3e-4,
          'k5': 5.8e-3, 'k6': 5.8e-3, 'k7': 1.73e-2, 'k8': 1.16e-2,
          'k9': 3.9e-3, 'k10': 3.9e-3, 'k11': 5e-4, 'k12': 1e-3,
          'k13': 1.16e-2, 'l3': 0.21, 'l8': 464, 'l9': 507,
          'l10': 81.9, 'l11': 0.21, 'l12': 440}

# Define full ODE system
dx1 = -k2*x1*x4 - k9*x1 + l9
dx2 = k2*x1*x4 - k5*x2 - k11*x2*x7 + l11*x8
dx3 = -k1*x2*x3 - k10*x3 + l10
dx4 = k1*x2*x3 - k3*x4*x5 - k6*x4 + l3*x6
dx5 = -k3*x4*x5 - k4*x4*x5 - k8*x5 + l3*x6 + l8
dx6 = k3*x4*x5 - k7*x6 - l3*x6
dx7 = -k11*x2*x7 - k12*x7 + l11*x8 + l12
dx8 = k11*x2*x7 - k13*x8 - l11*x8

odes = [dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8]

# Symbolically eliminate x1, x3, x5, x6, x7, x8 for 2D model
elims = sp.solve([dx1, dx3, dx5, dx6, dx7, dx8], [x1, x3, x5, x6, x7, x8], dict=True)
dx2_red = dx2.subs(elims[0])
dx4_red = dx4.subs(elims[0])

# Try solving dx2 = 0 for x2
x2_solutions = sp.solve(dx2_red, x2)
real_x2_exprs = [s for s in x2_solutions if not s.has(sp.I)]
x2_expr = real_x2_exprs[0] if real_x2_exprs else None

# Substitute into dx4 for 1D model
dx4_1d = dx4_red.subs(x2, x2_expr) if x2_expr else None

# Exported objects
__all__ = [
    "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
    "vars_full", "params", "param_symbols",
    "odes", "dx2_red", "dx4_red", "dx4_1d"
]
