from operator import add

from dynd import nd, ndt

import matplotlib
import matplotlib.pyplot

from benchrun import Benchmark, median
from benchtime import Timer, CUDATimer

#size = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
size = [10, 100, 1000, 10000, 100000, 1000000, 10000000]

class UniformBenchmark(Benchmark):
  parameters = ('size',)
  size = size

  def __init__(self, cuda = False):
    Benchmark.__init__(self)
    self.cuda = cuda

  @median
  def run(self, size):
    if self.cuda:
      dst_tp = ndt.type('cuda_device[{} * float64]'.format(size))
    else:
      dst_tp = ndt.type('{} * float64'.format(size))
    dst = nd.empty(dst_tp)

    with CUDATimer() if self.cuda else Timer() as timer:
      nd.uniform(dst_tp = dst_tp)

    return timer.elapsed_time()

class NumPyUniformBenchmark(Benchmark):
  parameters = ('size',)
  size = size

  @median
  def run(self, size):
    import numpy as np

    with Timer() as timer:
      np.random.uniform(size = size)

    return timer.elapsed_time()

class PyCUDAUniformBenchmark(Benchmark):
  parameters = ('size',)
  size = size

  def __init__(self, gen):
    Benchmark.__init__(self)
    self.gen = gen

  @median
  def run(self, size):
    import numpy as np

    with CUDATimer() as timer:
      self.gen.gen_uniform(size, np.float64)

    return timer.elapsed_time()

if __name__ == '__main__':
  cuda = True

  benchmark = UniformBenchmark(cuda = cuda)
  benchmark.plot_result(loglog = True)

  benchmark = NumPyUniformBenchmark()
  benchmark.plot_result(loglog = True)

  if cuda:
    from pycuda import curandom

    benchmark = PyCUDAUniformBenchmark(curandom.XORWOWRandomNumberGenerator())
    benchmark.plot_result(loglog = True)

  matplotlib.pyplot.show()