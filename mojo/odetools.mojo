from Matrix import Matrix

struct ForwardEulerSolver[N: Int]:
    var a: Float64
    var b: Float64

    @always_inline("nodebug")
    fn __init__(inout self, a: Float64, b: Float64):
        self.a = a
        self.b = b

    @always_inline("nodebug")
    fn solve[func: fn(u: Matrix[DType.float64], t: Float64) capturing -> Matrix[DType.float64]](self, inout u: Matrix[DType.float64]) -> Matrix[DType.float64]:
        # Solve the ODE using Forward Euler
        # Given:
        #   func:: Function: The right hand side of the ODE
        #   (owned) u0:: Matrix: The initial condition
        # Return:
        #   u:: Matrix: The solution of the ODE

        var dt = (self.b - self.a) / (N + 1)

        for i in range(0, N + 1):
            u += dt * func(u, i * dt)

        return u