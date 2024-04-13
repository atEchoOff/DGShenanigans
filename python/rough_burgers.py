import numpy as np
from prelims import DGPrelims
from odetools import ForwardEulerSolver
from time import time

def f(u):
    return u ** 2 / 2

def u0(x):
    return np.exp(-10 * x ** 2)

def get_times(N, M):
    full_start = time()

    dg = DGPrelims(N, M, is_periodic=True)

    prelim_time = time() - full_start

    RHS = lambda u, t: dg.RHS(f, u)

    solver = ForwardEulerSolver(10 * 2048, (0, .7))

    u = u0(dg.x)

    ode_start = time()
    sol = list(solver.solve(RHS, u, pause_every=float('inf')))[-1]

    ode_time = time() - ode_start

    return np.array([prelim_time, ode_time])

N = 4
M = 2048

print("Prelim time, ODE time")
for M in [1, 64, 128, 256, 512, 1024, 2048]:
    print(get_times(N, M))