from tensor import Tensor
from python import Python
from Matrix import Matrix

@value
struct DGPrelims[n: Int, m: Int]:
    var np: PythonObject # numpy module
    var sp: PythonObject # scipy module

    var x: Matrix[DType.float64] # the abscissas
    var w: Matrix[DType.float64] # the weights
    var D: Matrix[DType.float64] # the differentiation matrix

    var L: Matrix[DType.float64]
    var normal: Matrix[DType.float64]

    var h: Float64
    var is_periodic: Bool

    fn __init__(inout self, a: Float64, b: Float64, is_periodic: Bool = False) raises:
        # Allocate everything first... mojo has issues allocating during computation
        self.sp = Python.import_module("scipy")
        self.np = Python.import_module("numpy")

        self.x = Matrix[DType.float64](n, m)
        self.w = Matrix[DType.float64](n, 1)
        self.D = Matrix[DType.float64](n, n)

        self.L = Matrix[DType.float64](n, 2)
        self.normal = Matrix[DType.float64](2, m)

        self.h = (b - a) / m
        self.is_periodic = is_periodic
        
        # Set the weights and abscissas from what we have now
        # First we determine the abscissas
        var P = self.sp.special.legendre(n - 1).deriv()

        # The roots are our inner nodes
        var roots = self.np.sort(self.np.roots(P))

        # And we put -1 and 1 on the ends to get our final abscissas
        var x = -Matrix[DType.float64].ones(1, 1) | Matrix[DType.float64](roots).T() | Matrix[DType.float64].ones(1, 1)
        
        # Now we determine the weights
        for i in range(n):
            self.w[i] = 2 / (n * (n - 1) * self.sp.special.eval_legendre(n - 1, x[i]).to_float64() ** 2)

        # Fill the differentiation matrix
        var DP = P.deriv()
        var DDP = DP.deriv()

        var Psi = Tensor[DType.float64](n)
        var DPsi = Tensor[DType.float64](n)
        var DDPsi = Tensor[DType.float64](n)

        for i in range(n):
            var xi = x[i]
            Psi[i] = ((xi ** 2 - 1) * P(xi)).to_float64()
            DPsi[i] = ((xi ** 2 - 1) * DP(xi) + 2 * xi * P(xi)).to_float64()
            DDPsi[i] = ((xi ** 2 - 1) * DDP(xi) + 4 * xi * DP(xi) + 2 * P(xi)).to_float64()

        for i in range(n):
            for j in range(n):
                # Evaluate l'_i(x_j) where l_i is the lagrange polynomial
                if i != j:
                    self.D[j, i] = 1 / (x[j] - x[i]) ** 2 / DPsi[i] * (DPsi[j] * (x[j] - x[i]) - Psi[j])
                else:
                    self.D[j, i] = DDPsi[i] / DPsi[i] / 2

        # Fill our abscissas
        for j in range(m):
            var a_subinterval = a + j * (b - a) / m
            var b_subinterval = a_subinterval + (b - a) / m

            self.x[:, j] = a_subinterval + (b_subinterval - a_subinterval) / 2 * (x + 1)

        # Now we need our L matrix
        self.L[0, 0] = 1
        self.L[-1, -1] = 1

        # and our normal
        self.normal.fill_wide[0, :](-1)
        self.normal.fill_wide[1, :](1)

    @always_inline("nodebug")
    fn RHS[func: fn(x: Float64) -> Float64](self, u: Matrix[DType.float64]) -> Matrix[DType.float64]:
        # Compute the right hand side of the DG equation, given f and u

        var uM = u[0, :] / u[-1, :]
        var uP = Matrix[DType.float64](0, 0) # temporary assignment
        if self.is_periodic:
            uP = (uM[1, :] >> 1) /
                     (uM[0, :] << 1)
        else:
            uP = (uM[0, 0] * Matrix[DType.float64].ones(1, 1)) | (uM[1, :-1])
            uP = uP / ((uM[0, 1:]) | (uM[1, -1] * Matrix[DType.float64].ones(1, 1)))

        var f_avg = .5 * (uM.apply[func]() + uP.apply[func]())
        
        return (-2 / self.h) * (self.D @ u.apply[func]() + Matrix[DType.float64].diag(1 / self.w) @ self.L @ ((f_avg - uM.apply[func]()) * self.normal - .5 * (uP - uM)))