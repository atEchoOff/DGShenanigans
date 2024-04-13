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

fn runme[N: Int, M: Int]() raises:
    var dg_start = now()
    var dg = DGPrelims[N, M](-1, 1, is_periodic = True)

    var dg_time = (now() - dg_start) * 1e-9

    # Define the right hand side for our ODE solver
    @always_inline("nodebug")
    @parameter
    fn RHS(u: Matrix[DType.float64], t: Float64) -> Matrix[DType.float64]:
        return dg.RHS[f](u)

    # Set up an ODE solver
    var solver = ForwardEulerSolver[10 * 2048](0, .7)

    var u = dg.x.apply[u0]()

    var ode_start = now()

    # Get the final solution
    var sol = solver.solve[RHS](u)

    var ode_time = (now() - ode_start) * 1e-9

    print(str(dg_time) + " " + str(ode_time))

fn main() raises:
    alias N = 4
    runme[N, 2048]()