from copy import deepcopy
import numpy as np

class ForwardEulerSolver:
    def __init__(self, Δt, time_domain):
        self.Δt = Δt
        self.time_domain = time_domain

    def solve(self, f, u0, pause_every=10):
        # Solve the ODE using forward Euler
        # Pause every (pause_every) frames to allow user to perform actions on
        # the current solution

        u = deepcopy(u0)

        for i, t in enumerate(np.arange(*self.time_domain, self.Δt)):
            if i % pause_every == 0:
                yield u, t

            u += self.Δt * f(u, t)