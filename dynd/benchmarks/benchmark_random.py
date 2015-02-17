from operator import add

import numpy as np

from dynd import nd, ndt

import matplotlib
import matplotlib.pyplot

from benchrun import Benchmark, median
from benchtime import Clock, NumPyClock, PyCUDAClock

n = 10
#size = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
size = [10, 100, 1000, 10000, 100000, 1000000, 10000000]

class UniformBenchmark(Benchmark):
  parameters = ('size',)
  size = size

  def __init__(self, cuda = False):
    Benchmark.__init__(self)
    self.cuda = cuda

  @median(n)
  def run(self, size):
    if self.cuda:
      dst_tp = ndt.type('cuda_device[{} * float64]'.format(size))
    else:
      dst_tp = ndt.type('{} * float64'.format(size))
    dst = nd.empty(dst_tp)

    with Clock(self.cuda) as clock:
      nd.uniform(dst_tp = dst_tp)

    return clock.elapsed()

class NumPyUniformBenchmark(Benchmark):
  parameters = ('size',)
  size = size

  @median(n)
  def run(self, size):
    import numpy as np

    with NumPyClock() as clock:
      np.random.uniform(size = size)

    return clock.elapsed()

class PyCUDAUniformBenchmark(Benchmark):
  parameters = ('size',)
  size = size

  def __init__(self, gen):
    Benchmark.__init__(self)
    self.gen = gen

  @median(n)
  def run(self, size):
    with PyCUDAClock() as clock:
      self.gen.gen_uniform(size, np.float64)

    return clock.elapsed()

if __name__ == '__main__':
  cuda = True

  benchmark = UniformBenchmark(cuda = cuda)
  benchmark.plot_result(loglog = True)

  benchmark = NumPyUniformBenchmark()
  benchmark.plot_result(loglog = True)

  if cuda:
    from pycuda import autoinit, curandom

    benchmark = PyCUDAUniformBenchmark(curandom.XORWOWRandomNumberGenerator())
    benchmark.plot_result(loglog = True)

  matplotlib.pyplot.show()