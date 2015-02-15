import numpy as np

from dynd import nd, ndt

from benchrun import Benchmark, clock, mean

n = 100
size = [10, 100, 1000, 10000, 100000]

class ArithmeticBenchmark(Benchmark):
  parameters = ('size',)
  size = size

  @mean(n)
  def run(self, size):
    a = nd.uniform(dst_tp = ndt.type('{} * float64'.format(size)))
    b = nd.uniform(dst_tp = ndt.type('{} * float64'.format(size)))

    start = clock()
    a + b
    stop = clock()

    return stop - start

class NumPyArithmeticBenchmark(Benchmark):
  parameters = ('size',)
  size = size

  @mean(n)
  def run(self, size):
    a = np.random.uniform(size = size)
    b = np.random.uniform(size = size)

    start = clock()
    a + b
    stop = clock()

    return stop - start

if __name__ == '__main__':
  import matplotlib
  import matplotlib.pyplot

  benchmark = ArithmeticBenchmark()
  benchmark.plot_result(loglog = True)

  benchmark = NumPyArithmeticBenchmark()
  benchmark.plot_result(loglog = True)

  matplotlib.pyplot.show()