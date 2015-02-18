from operator import add

from dynd import nd, ndt

import matplotlib
import matplotlib.pyplot

from benchrun import Benchmark, median
from benchtime import Timer, CUDATimer

size = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
#size = [10, 100, 1000, 10000, 100000]

class ArithmeticBenchmark(Benchmark):
  parameters = ('size',)
  size = size

  def __init__(self, op, cuda = False):
    Benchmark.__init__(self)
    self.op = op
    self.cuda = cuda

  @median
  def run(self, size):
    if self.cuda:
      dst_tp = ndt.type('cuda_device[{} * float64]'.format(size))
    else:
      dst_tp = ndt.type('{} * float64'.format(size))

    a = nd.uniform(dst_tp = dst_tp)
    b = nd.uniform(dst_tp = dst_tp)

    with CUDATimer() if self.cuda else Timer() as timer:
      self.op(a, b)

    return timer.elapsed_time()

class NumPyArithmeticBenchmark(Benchmark):
  parameters = ('size',)
  size = size

  def __init__(self, op):
    Benchmark.__init__(self)
    self.op = op

  @median
  def run(self, size):
    import numpy as np

    a = np.random.uniform(size = size)
    b = np.random.uniform(size = size)

    with Timer() as timer:
      self.op(a, b)

    return timer.elapsed_time()

class PyCUDAArithmeticBenchmark(Benchmark):
  parameters = ('size',)
  size = size

  def __init__(self, op):
    Benchmark.__init__(self)
    self.op = op

  @median
  def run(self, size):
    import numpy as np
    from pycuda import curandom

    a = curandom.rand(size, dtype = np.float64)
    b = curandom.rand(size, dtype = np.float64)

    with CUDATimer() as timer:
      self.op(a, b)

    return timer.elapsed_time()

if __name__ == '__main__':
  cuda = True

  benchmark = ArithmeticBenchmark(add, cuda = False)
  benchmark.plot_result(loglog = True)

  benchmark = NumPyArithmeticBenchmark(add)
  benchmark.plot_result(loglog = True)

#  if cuda:
 #   benchmark = PyCUDAArithmeticBenchmark(add)
  #  benchmark.plot_result(loglog = True)

  matplotlib.pyplot.show()