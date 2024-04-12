from copy import deepcopy
import numpy as np

class ForwardEulerSolver:
    def __init__(self, N, time_domain):
        self.N = N
        self.time_domain = time_domain

    def solve(self, f, u0, pause_every=10):
        # Solve the ODE using Forward Euler
        # Given:
        #   f:: Function: The right hand side of the ODE
        #   u0:: np.array: The initial condition
        #   pause_every:: int: The number of frames to skip before pausing (default = 10)
        # Return:
        #   Generator: The solution to the ODE

        u = deepcopy(u0)

        Δt = (self.time_domain[1] - self.time_domain[0]) / (self.N + 1)

        for i, t in enumerate(np.linspace(*self.time_domain, self.N + 1)):
            if i % pause_every == 0:
                yield u, t

            u += Δt * f(u, t)

        yield u, t
    
class RK4Solver:
    def __init__(self, N, time_domain):
        self.N = N
        self.time_domain = time_domain

    def solve(self, f, u0, pause_every=10):
        # Solve the ODE using RK4
        # Given:
        #   f:: Function: The right hand side of the ODE
        #   u0:: np.array: The initial condition
        #   pause_every:: int: The number of frames to skip before pausing (default = 10)
        # Return:
        #   Generator: The solution to the ODE

        u = deepcopy(u0)

        Δt = (self.time_domain[1] - self.time_domain[0]) / (self.N + 1)

        for i, t in enumerate(np.linspace(*self.time_domain, self.N + 1)):
            if i % pause_every == 0:
                yield u, t

            k1 = Δt * f(u, t)
            k2 = Δt * f(u + 0.5 * k1, t + 0.5 * Δt)
            k3 = Δt * f(u + 0.5 * k2, t + 0.5 * Δt)
            k4 = Δt * f(u + k3, t + Δt)

            u += (k1 + 2 * k2 + 2 * k3 + k4) / 6

        yield u, t