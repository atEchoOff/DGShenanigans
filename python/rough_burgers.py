import numpy as np
from prelims import DGPrelims
from odetools import ForwardEulerSolver
from time import time

def f(u):
    return u ** 2 / 2

def u0(x):
    return np.exp(-10 * x ** 2)

N = 4
M = 2048

full_start = time()

dg = DGPrelims(N, M, is_periodic=False)

RHS = lambda u, t: dg.RHS(f, u)

solver = ForwardEulerSolver(50 * M, (0, .7))

u = u0(dg.x)

ode_start = time()
sol = list(solver.solve(RHS, u, pause_every=float('inf')))[-1]

print("ODE Time elapsed: ", time() - ode_start)
print(f"Full time: {time() - full_start}")