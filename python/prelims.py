from scipy.special import legendre, eval_legendre
import numpy as np

def gausslob(n):
    # First we determine the abscissas
    P = legendre(n - 1).deriv()

    # The roots are our inner nodes
    roots = np.sort(np.roots(P))

    # And we put -1 and 1 on the ends to get our final abscissas
    x = np.concatenate(([-1], roots, [1]))

    # Now we determine the weights
    w = np.zeros(n)
    for i in range(n):
        w[i] = 2 / (n * (n - 1) * eval_legendre(n - 1, x[i]) ** 2)

    return x, w, P

def diffmatr(x, P, n):
    # Return the differentiation matrix
    D = np.zeros((n, n))

    DP = P.deriv()
    DDP = DP.deriv()

    Ψ = [(xi ** 2 - 1) * P(xi) for xi in x]
    DΨ = [(xi ** 2 - 1) * DP(xi) + 2 * xi * P(xi) for xi in x]
    DDΨ = [(xi ** 2 - 1) * DDP(xi) + 4 * xi * DP(xi) + 2 * P(xi) for xi in x]

    for i in range(n):
        for j in range(n):
            # Evaluate l'_i(x_j) where l_i is the lagrange polynomial
            if i != j:
                D[j, i] = 1 / (x[j] - x[i]) ** 2 / DΨ[i] * (DΨ[j] * (x[j] - x[i]) - Ψ[j])
            else:
                D[j, i] = DDΨ[i] / DΨ[i] / 2

    return D

def expand(x, m, interval):
    # Expand mesh points x over [-1, 1] to a series of nodes over [a, b] = interval
    # where there are m total subintervals
    a, b = interval
    ret = np.zeros((x.shape[0], m))

    for j in range(m):
        a_subinterval = a + j * (b - a) / m
        b_subinterval = a_subinterval + (b - a) / m

        ret[:, j] = a_subinterval + (b_subinterval - a_subinterval) / 2 * (x + 1)

    return ret

def L(n):
    # Generate L matrix
    L = np.zeros((n, 2))
    L[0, 0] = 1
    L[-1, -1] = 1

    return L

def normal(m):
    # Generate the normal matrix
    return np.vstack((-np.ones(m), np.ones(m)))


class DGPrelims:
    def __init__(self, n, m, interval=(-1, 1), is_periodic=False):
        self.n = n
        self.m = m
        self.x, self.w, self.P = gausslob(n)
        self.D = diffmatr(self.x, self.P, n)

        self.x = expand(self.x, m, interval)

        self.L = L(n)
        self.normal = normal(m)
        self.h = (interval[1] - interval[0]) / m

        self.is_periodic = is_periodic

    def RHS(self, f, u):
        # Compute the right hand side of the DG equation, given f and u
        
        uM = u[[0, self.n - 1], :]
        if self.is_periodic:
            uP = np.vstack((np.roll(uM[1, :], 1), np.roll(uM[0, :], -1)))
        else:
            uP = np.concatenate((np.array([uM[0, 0]]), uM[1, :-1]))
            uP = np.vstack((uP, np.concatenate((uM[0, 1:], [uM[1, -1]]))))

        f_avg = 0.5 * (f(uM) + f(uP))

        ret = (-2 / self.h) * (self.D @ f(u) + np.diag(1 / self.w) @ self.L @ ((f_avg - f(uM)) * self.normal - 0.5 * (uP - uM)))
        
        return ret
    
    def integrate(self, U):
        # Compute the estimated integral of U over the domain, using the Gauss Lobatto quadrature
        integral = 0
        for m in range(0, self.m):
            integral += np.sum(self.w * U[:,m])

        return integral * self.h / 2 # FIXME why 1 / 2?

# Testing stuff
if __name__ == "__main__":
    from time import time

    start = time()
    dg = DGPrelims(4, 16384, is_periodic=True)

    print("Time to compute DG Preliminaries: ", time() - start)

    def f(u):
        return u ** 2 / 2
    
    u = dg.x

    start2 = time()
    result = dg.RHS(f, u)

    print("Time elapsed: ", time() - start2)

    # print(result)