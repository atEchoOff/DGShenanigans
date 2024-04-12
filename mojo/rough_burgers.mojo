from prelims import DGPrelims
from odetools import ForwardEulerSolver
from math import exp
from Matrix import Matrix
from time import now
from python import Python

# Define some burger functions
@always_inline("nodebug")
fn f(u: Float64) -> Float64:
    return u ** 2 / 2

fn u0(x: Float64) -> Float64:
    return exp(-10 * x ** 2)

fn main() raises:
    alias N = 4
    alias M = 256
    var full_start = now()
    var dg = DGPrelims[N, M](-1, 1)

    # Define the right hand side for our ODE solver
    @parameter
    fn RHS(u: Matrix[DType.float64], t: Float64) -> Matrix[DType.float64]:
        return dg.RHS[f](u)

    # Set up an ODE solver
    var solver = ForwardEulerSolver[50 * M](0, .7)

    var u = dg.x.apply[u0]()

    var start2 = now()

    # Get the final solution
    var sol = solver.solve[RHS](u)

    print("Time taken full: ", (now() - full_start) * 1e-9)
    print("Time taken ode: ", (now() - start2) * 1e-9)

    # Lets see what it looks like!
    var plt = Python.import_module("matplotlib.pyplot")

    plt.plot(~dg.x, ~sol)
    plt.savefig("burger.png")