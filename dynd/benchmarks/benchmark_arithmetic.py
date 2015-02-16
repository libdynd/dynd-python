from operator import add

import numpy as np

try:
  import pycuda
  import pycuda.autoinit
  import pycuda.curandom
except ImportError:
  pass

from dynd import nd, ndt

import matplotlib
import matplotlib.pyplot

from benchrun import Benchmark, clock, mean, median

n = 10
#size = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
size = [10, 100, 1000, 10000, 100000]

class ArithmeticBenchmark(Benchmark):
  parameters = ('size',)
  size = size

  def __init__(self, op, cuda = False):
    Benchmark.__init__(self)
    self.op = op
    self.cuda = cuda

  @median(n)
  def run(self, size):
    if self.cuda:
      dst_tp = ndt.type('cuda_device[{} * float64]'.format(size))
    else:
      dst_tp = ndt.type('{} * float64'.format(size))

    a = nd.uniform(dst_tp = dst_tp)
    b = nd.uniform(dst_tp = dst_tp)

    self.op(a, b)
    start = clock()
    self.op(a, b)
    stop = clock()

    return stop - start

class NumPyArithmeticBenchmark(Benchmark):
  parameters = ('size',)
  size = size

  def __init__(self, op):
    Benchmark.__init__(self)
    self.op = op

  @median(n)
  def run(self, size):
    a = np.random.uniform(size = size)
    b = np.random.uniform(size = size)

    self.op(a, b)
    start = clock()
    self.op(a, b)
    stop = clock()

    return stop - start

class PyCUDAArithmeticBenchmark(Benchmark):
  parameters = ('size',)
  size = size

  def __init__(self, op):
    Benchmark.__init__(self)
    self.op = op

  @mean(n)
  def run(self, size):
    a = pycuda.curandom.rand(size, dtype = np.float64)
    b = pycuda.curandom.rand(size, dtype = np.float64)

    self.op(a, b)
    start = clock()
    self.op(a, b)
    stop = clock()

    return stop - start

if __name__ == '__main__':
  cuda = True

  benchmark = ArithmeticBenchmark(add, cuda = cuda)
  benchmark.plot_result(loglog = True)

  if (not cuda):
    benchmark = NumPyArithmeticBenchmark(add)
    benchmark.plot_result(loglog = True)

  if cuda:
    benchmark = PyCUDAArithmeticBenchmark(add)
    benchmark.plot_result(loglog = True)

  matplotlib.pyplot.show()