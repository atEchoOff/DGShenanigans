from tensor import Tensor
from utils.index import Index
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from algorithm import vectorize, parallelize
from math import sqrt
from python import Python, PythonObject
from buffer import parallel_memcpy
from algorithm import Static2DTileUnitFunc as Tile2DFunc

# Perform 2D tiling on the iteration space defined by end_x and end_y.
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    # Note: this assumes that ends are multiples of the tiles.
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)

struct Matrix[dtype: DType]:
    var m: Int
    var n: Int
    var data: DTypePointer[dtype]

    @always_inline("nodebug")
    fn __init__(inout self, m: Int, n: Int):
        # Create a matrix with zeros
        self.m = m
        self.n = n
        self.data = DTypePointer[dtype].alloc(m * n)

        memset_zero(self.data, m * n)

    @always_inline("nodebug")
    fn __init__(inout self, data: Tensor[dtype]):
        self.m = data.shape()[0]
        self.n = data.shape()[1]
        self.data = DTypePointer[dtype].alloc(self.m * self.n)
        for i in range(0, self.m):
            for j in range(0, self.n):
                self.data[i * self.n + j] = data[Index(i, j)]

    @always_inline("nodebug")
    fn __init__(inout self, m: Int, n: Int, data: DTypePointer[dtype]):
        self.m = m
        self.n = n
        self.data = data

    @always_inline("nodebug")
    fn __init__(inout self, data: List[List[SIMD[dtype, 1]]]):
        self.m = len(data)
        self.n = len(data[0])
        self.data = DTypePointer[dtype].alloc(self.m * self.n)

        for i in range(0, self.m):
            for j in range(0, self.n):
                self.data[i * self.n + j] = data[i][j]

    @always_inline
    fn __init__(inout self, nparr: PythonObject):
        # Obtain data from numpy array
        try:
            if len(nparr.shape) == 2:
                # This is a matrix
                self.m = int(nparr.shape[0])
                self.n = int(nparr.shape[1])

                self.data = DTypePointer[dtype].alloc(self.m * self.n)
                for i in range(0, self.m):
                    for j in range(0, self.n):
                        self.data[i * self.n + j] = nparr[(i, j)].to_float64().cast[dtype]()
            else:
                # This is a vector
                self.m = int(nparr.size)
                self.n = 1
                self.data = DTypePointer[dtype].alloc(self.m)

                for i in range(0, self.m):
                    self.data[i] = nparr[i].to_float64().cast[dtype]()
        except e:
            print(e)
            self.m = 0
            self.n = 0
            self.data = Tensor[dtype]()._ptr

    fn __del__(owned self):
        return self.data.free()

    @always_inline("nodebug")
    @staticmethod
    fn ones(m: Int, n: Int) -> Matrix[dtype]:
        var data = DTypePointer[dtype].alloc(m * n)

        for i in range(0, m):
            for j in range(0, n):
                data[i * n + j] = 1

        return Matrix[dtype](m, n, data)

    @always_inline("nodebug")
    @staticmethod
    fn diag(data: Matrix[dtype]) -> Matrix[dtype]:
        var ret = DTypePointer[dtype].alloc(data.m * data.m)
        memset_zero(ret, data.m * data.m)

        for i in range(0, data.m):
            ret[i * data.m + i] = data.data[i]

        return Matrix[dtype](data.m, data.m, ret)

    @always_inline("nodebug")
    fn __copyinit__(inout self, other: Matrix[dtype]):
        self.m = other.m
        self.n = other.n
        self.data = DTypePointer[dtype].alloc(self.m * self.n)
        parallel_memcpy[dtype](self.data, other.data, self.m * self.n)
        

    @always_inline("nodebug")
    fn size_str(self) -> String:
        return "[" + str(self.m) + ", " + str(self.n) + "]"

    @always_inline("nodebug")
    fn fix_idx_i(self, i: Int) -> Int:
        if i < 0:
            return self.m + i
        return i

    @always_inline("nodebug")
    fn fix_slice_i(self, slice: Slice) -> Slice:
        var ret = Slice(slice.start, slice.end, slice.step)
        if slice.start < 0:
            ret.start = self.m + slice.start
        if slice.end < 0:
            ret.end = self.m + slice.end
        elif Int32(slice.end) == Int32.MAX:
            ret.end = self.m

        return ret

    @always_inline("nodebug")
    fn fix_idx_j(self, j: Int) -> Int:
        if j < 0:
            return self.n + j
        return j

    @always_inline("nodebug")
    fn fix_slice_j(self, slice: Slice) -> Slice:
        var ret = Slice(slice.start, slice.end, slice.step)
        if slice.start < 0:
            ret.start = self.n + slice.start
        if slice.end < 0:
            ret.end = self.n + slice.end
        elif Int32(slice.end) == Int32.MAX:
            ret.end = self.n

        return ret

    @always_inline("nodebug")
    fn __matmul__(self, other: Matrix[dtype]) -> Matrix[dtype]:
        # Simple, FIXME extremely naive matrix multiplication
        # also very unsafe, it is the user's responsibility to ensure dimensions match
        var ret = DTypePointer[dtype].alloc(self.m * other.n)
        memset_zero(ret, self.m * other.n)

        for i in range(0, self.m):
            for k in range(0, self.n):
                for j in range(0, other.n):
                    ret[i * other.n + j] += self.data[i * self.n + k] * other.data[k * other.n + j]

        return Matrix[dtype](self.m, other.n, ret)

    # @always_inline("nodebug")
    # fn __matmul__(self, other: Matrix[dtype]) -> Matrix[dtype]:
    #     var C = DTypePointer[dtype].alloc(self.m * other.n)
    #     @parameter
    #     fn calc_row(m: Int):
    #         @parameter
    #         fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
    #             for k in range(y, y + tile_y):
    #                 @parameter
    #                 fn dot[nelts: Int](n: Int):
    #                     C.store[width=nelts](m * other.n + n + x, C.load[width=nelts](m * other.n + n + x) + self[m, k] * other.data.load[width=nelts](k * other.n + n + x))

    #                 # Vectorize by nelts and unroll by tile_x/nelts
    #                 # Here unroll factor is 4
    #                 alias unroll_factor = tile_x // 4
    #                 vectorize[dot, 4, size=tile_x, unroll_factor=unroll_factor]()

    #         alias tile_size = 4
    #         tile[calc_tile, 4 * tile_size, tile_size](self.n, other.n)

    #     parallelize[calc_row](self.m, self.m)

    #     return Matrix[dtype](self.m, other.n, C)
    
    @always_inline("nodebug")
    fn __mul__(self, other: SIMD[dtype, 1]) -> Matrix[dtype]:
        var newdata = DTypePointer[dtype].alloc(self.m * self.n)

        for i in range(0, self.m):
            for j in range(0, self.n):
                newdata[i * self.n + j] = self.data[i * self.n + j] * other

        return Matrix[dtype](self.m, self.n, newdata)

    @always_inline("nodebug")
    fn __rmul__(self, other: SIMD[dtype, 1]) -> Matrix[dtype]:
        var newdata = DTypePointer[dtype].alloc(self.m * self.n)

        for i in range(0, self.m):
            for j in range(0, self.n):
                newdata[i * self.n + j] = self.data[i * self.n + j] * other

        return Matrix[dtype](self.m, self.n, newdata)

    @always_inline("nodebug")
    fn __truediv__(self, other: SIMD[dtype, 1]) -> Matrix[dtype]:
        var newdata = DTypePointer[dtype].alloc(self.m * self.n)

        for i in range(0, self.m):
            for j in range(0, self.n):
                newdata[i * self.n + j] = self.data[i * self.n + j] / other

        return Matrix[dtype](self.m, self.n, newdata)

    @always_inline("nodebug")
    fn __rtruediv__(self, other: SIMD[dtype, 1]) -> Matrix[dtype]:
        var newdata = DTypePointer[dtype].alloc(self.m * self.n)

        for i in range(0, self.m):
            for j in range(0, self.n):
                newdata[i * self.n + j] = other / self.data[i * self.n + j]

        return Matrix[dtype](self.m, self.n, newdata)

    @always_inline("nodebug")
    fn __getitem__(self, i: Int, j: Int) -> SIMD[dtype, 1]:
        return self.data[self.fix_idx_i(i) * self.n + self.fix_idx_j(j)]

    @always_inline("nodebug")
    fn __getitem__(self, i: Int) -> SIMD[dtype, 1]:
        return self.data[self.fix_idx_i(i)]
    
    @always_inline("nodebug")
    fn __getitem__(self, i: Int, jslice: Slice) -> Matrix[dtype]:
        var ii = self.fix_idx_i(i)
        var jsl = self.fix_slice_j(jslice)

        var ret = DTypePointer[dtype].alloc(jsl.__len__())

        for j in range(jsl.__len__()):
            ret[j] = self.data[ii * self.n + jsl[j]]

        return Matrix[dtype](1, jsl.__len__(), ret)

    @always_inline("nodebug")
    fn __getitem__(self, islice: Slice, j: Int) -> Matrix[dtype]:
        var isl = self.fix_slice_i(islice)
        var jj = self.fix_idx_j(j)

        var ret = DTypePointer[dtype].alloc(isl.__len__())

        for i in range(isl.__len__()):
            ret[i] = self.data[isl[i] * self.n + jj]

        return Matrix[dtype](isl.__len__(), 1, ret)

    @always_inline("nodebug")
    fn __getitem__(self, islice: Slice, jslice: Slice) -> Matrix[dtype]:
        var isl = self.fix_slice_i(islice)
        var jsl = self.fix_slice_j(jslice)

        var ret = DTypePointer[dtype].alloc(isl.__len__() * jsl.__len__())

        var new_height = isl.__len__()

        for i in range(isl.__len__()):
            for j in range(jsl.__len__()):
                ret[i * new_height + j] = self.data[isl[i] * self.n + jsl[j]]

        return Matrix[dtype](new_height, jsl.__len__(), ret)

    @always_inline("nodebug")
    fn __setitem__(inout self, i: Int, j: Int, val: SIMD[dtype, 1]):
        self.data[self.fix_idx_i(i) * self.n + self.fix_idx_j(j)] = val

    @always_inline("nodebug")
    fn __setitem__(inout self, i: Int, val: SIMD[dtype, 1]):
        self.data[self.fix_idx_i(i)] = val

    @always_inline("nodebug")
    fn __setitem__(inout self, i: Int, jslice: Slice, val: Matrix[dtype]):
        var ii = self.fix_idx_i(i)
        var jsl = self.fix_slice_j(jslice)

        for j in range(jsl.__len__()):
            self.data[ii * self.n + jsl[j]] = val.data[j]

    @always_inline("nodebug")
    fn fill_wide[i: Int, jslice: Slice](inout self, val: SIMD[dtype, 1]):
        var ii = self.fix_idx_i(i)
        var jsl = self.fix_slice_j(jslice)

        for j in range(jsl.__len__()):
            self.data[ii * self.n + jsl[j]] = val

    @always_inline("nodebug")
    fn __setitem__(inout self, islice: Slice, j: Int, val: Matrix[dtype]):
        var isl = self.fix_slice_i(islice)
        var jj = self.fix_idx_j(j)

        for i in range(isl.__len__()):
            self.data[isl[i] * self.n + jj] = val.data[i]

    @always_inline("nodebug")
    fn fill_tall[islice: Slice, j: Int](inout self, val: SIMD[dtype, 1]):
        var isl = self.fix_slice_i(islice)
        var jj = self.fix_idx_j(j)

        for i in range(isl.__len__()):
            self.data[isl[i] * self.n + jj] = val

    @always_inline("nodebug")
    fn __setitem__(inout self, islice: Slice, jslice: Slice, val: Matrix[dtype]):
        var isl = self.fix_slice_i(islice)
        var jsl = self.fix_slice_j(jslice)

        var new_height = isl.__len__()

        for i in range(isl.__len__()):
            for j in range(jsl.__len__()):
                self.data[isl[i] * self.n + jsl[j]] = val.data[i * new_height + j]

    @always_inline("nodebug")
    fn fill_block[islice: Slice, jslice: Slice](inout self, val: SIMD[dtype, 1]):
        var isl = self.fix_slice_i(islice)
        var jsl = self.fix_slice_j(jslice)

        for i in range(isl.__len__()):
            for j in range(jsl.__len__()):
                self.data[isl[i] * self.n + jsl[j]] = val

    @always_inline("nodebug")
    fn __add__(self, other: Matrix[dtype]) -> Matrix[dtype]:
        var val = DTypePointer[dtype].alloc(self.m * self.n)

        for i in range(0, self.m):
            for j in range(0, self.n):
                val[i * self.n + j] = self.data[i * self.n + j] + other.data[i * self.n + j]
                # self.data.store[width=8](i * self.n + j, self.data.load[width=8](i * self.n + j) + other.data.load[width=8](i * self.n + j))

        return Matrix[dtype](self.m, self.n, val)

    @always_inline("nodebug")
    fn __iadd__(inout self, other: Matrix[dtype]):
        for i in range(0, self.m):
            for j in range(0, self.n):
                self.data[i * self.n + j] += other.data[i * self.n + j]
                # self.data.store[width=8](i * self.n + j, self.data.load[width=8](i * self.n + j) + other.data.load[width=8](i * self.n + j))

        # @parameter
        # fn vecmath[simdwidth: Int](idx: Int)->None:
        #     self.data.store[width=simdwidth](idx, self.data.load[width=simdwidth](idx) + other.data.load[width=simdwidth](idx))

        # vectorize[vecmath, 512, size = 4]()

        # @parameter
        # fn parmath(idx: Int)->None:
        #     self.data.store[width=512](idx, self.data.load[width=512](idx) + other.data.load[width=512](idx))

        # parallelize[parmath](4)

    @always_inline("nodebug")
    fn __add__(self, other: SIMD[dtype, 1]) -> Matrix[dtype]:
        var ret = DTypePointer[dtype].alloc(self.m * self.n)

        for i in range(0, self.m):
            for j in range(0, self.n):
                ret[i * self.n + j] = self.data[i * self.n + j] + other

        return Matrix[dtype](self.m, self.n, ret)

    @always_inline("nodebug")
    fn __radd__(self, other: SIMD[dtype, 1]) -> Matrix[dtype]:
        var ret = DTypePointer[dtype].alloc(self.m * self.n)

        for i in range(0, self.m):
            for j in range(0, self.n):
                ret[i * self.n + j] = self.data[i * self.n + j] + other

        return Matrix[dtype](self.m, self.n, ret)

    @always_inline("nodebug")
    fn __sub__(self, other: Matrix[dtype]) -> Matrix[dtype]:
        var val = DTypePointer[dtype].alloc(self.m * self.n)

        for i in range(0, self.m):
            for j in range(0, self.n):
                val[i * self.n + j] = self.data[i * self.n + j] - other.data[i * self.n + j]

        return Matrix[dtype](self.m, self.n, val)

    @always_inline("nodebug")
    fn __sub__(self, other: SIMD[dtype, 1]) -> Matrix[dtype]:
        var ret = DTypePointer[dtype].alloc(self.m * self.n)

        for i in range(0, self.m):
            for j in range(0, self.n):
                ret[i * self.n + j] = self.data[i * self.n + j] - other

        return Matrix[dtype](self.m, self.n, ret)

    @always_inline("nodebug")
    fn __rsub__(self, other: SIMD[dtype, 1]) -> Matrix[dtype]:
        var ret = DTypePointer[dtype].alloc(self.m * self.n)

        for i in range(0, self.m):
            for j in range(0, self.n):
                ret[i * self.n + j] = other - self.data[i * self.n + j]

        return Matrix[dtype](self.m, self.n, ret)

    @always_inline("nodebug")
    fn __mul__(self, other: Matrix[dtype]) -> Matrix[dtype]:
        var val = DTypePointer[dtype].alloc(self.m * self.n)

        for i in range(0, self.m):
            for j in range(0, self.n):
                val[i * self.n + j] = self.data[i * self.n + j] * other.data[i * self.n + j]

        return Matrix[dtype](self.m, self.n, val)

    @always_inline("nodebug")
    fn __neg__(self) -> Matrix[dtype]:
        return -1 * self

    @always_inline("nodebug")
    fn __str__(self) -> String:
        var ret: String = ""
        for i in range(0, self.m):
            for j in range(0, self.n):
                ret += str(self.data[i * self.n + j]) + "\t"
            ret += "\n"
        return ret
    
    @always_inline("nodebug")
    fn apply[func: fn(x: SIMD[dtype, 1]) -> SIMD[dtype, 1]](self) -> Matrix[dtype]:
        var ret = DTypePointer[dtype].alloc(self.m * self.n)

        for i in range(0, self.m):
            for j in range(0, self.n):
                ret[i * self.n + j] = func(self.data[i * self.n + j])

        return Matrix[dtype](self.m, self.n, ret)

    @always_inline("nodebug")
    fn __or__(self, other: Matrix[dtype]) -> Matrix[dtype]:
        var ret = DTypePointer[dtype].alloc(self.m * (self.n + other.n))

        for i in range(0, self.m):
            for j in range(0, self.n):
                ret[i * (self.n + other.n) + j] = self.data[i * other.n + j]

            for j in range(0, other.n):
                ret[i * (self.n + other.n) + self.n + j] = other.data[i * other.n + j]

        return Matrix[dtype](self.m, self.n + other.n, ret)

    @always_inline("nodebug")
    fn __truediv__(self, other: Matrix[dtype]) -> Matrix[dtype]:
        var ret = DTypePointer[dtype].alloc((self.m + other.m) * self.n)

        for i in range(0, self.m):
            for j in range(0, self.n):
                ret[i * self.n + j] = self.data[i * other.n + j]

        for i in range(0, other.m):
            for j in range(0, self.n):
                ret[(i + self.m) * self.n + j] = other.data[i * other.n + j]

        return Matrix[dtype](self.m + other.m, self.n, ret)

    @always_inline("nodebug")
    fn T(self) -> Matrix[dtype]:
        # FIXME this is very inefficient
        var ret = Tensor[dtype](self.n, self.m)

        for i in range(0, self.m):
            for j in range(0, self.n):
                ret[j * self.m + i] = self.data[i * self.n + j]

        return Matrix[dtype](ret)

    @always_inline("nodebug")
    fn __lshift__(self, val: Int) -> Matrix[dtype]:
        var ret = DTypePointer[dtype].alloc(self.m * self.n)

        for i in range(0, self.m):
            for j in range(0, self.n):
                ret[i * self.n +  ((j + self.n - val) % self.n)] = self.data[i * self.n + j]

        return Matrix[dtype](self.m, self.n, ret)

    @always_inline("nodebug")
    fn __rshift__(self, val: Int) -> Matrix[dtype]:
        var ret = DTypePointer[dtype].alloc(self.m * self.n)

        for i in range(0, self.m):
            for j in range(0, self.n):
                ret[i * self.n +  ((j + val) % self.n)] = self.data[i * self.n + j]

        return Matrix[dtype](self.m, self.n, ret)

    @always_inline("nodebug")
    fn frobenius(self) -> SIMD[dtype, 1]:
        var ret = SIMD[dtype, 1](0)
        for i in range(0, self.m):
            for j in range(0, self.n):
                ret += self.data[i * self.n + j] ** 2

        return sqrt(ret)

    @always_inline("nodebug")
    fn __invert__(self) -> PythonObject:
        # Convert to numpy array
        # Dont ask me why I just really like
        # the syntax :)
        try:
            var np = Python.import_module("numpy")
            var nparr = np.zeros((self.m, self.n), dtype=np.float64)

            for i in range(0, self.m):
                for j in range(0, self.n):
                    nparr.itemset((i, j), self.data[i * self.n + j])

            return nparr
        except:
            return None