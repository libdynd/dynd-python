import numpy as np

from dynd import nd, ndt

from benchrun import Benchmark, clock

class ArithmeticBenchmark(Benchmark):
  parameters = ('size',)
  size = [100000, 10000000, 100000000]

  def run(self, size):
    a = nd.uniform(dst_tp = ndt.type('{} * float64'.format(size)))
    b = nd.uniform(dst_tp = ndt.type('{} * float64'.format(size)))

    start = clock()
    a + b
    stop = clock()

    return stop - start

class NumPyArithmeticBenchmark(Benchmark):
  parameters = ('size',)
  size = [100000, 10000000, 100000000]

  def run(self, size):
    a = np.random.uniform(size = size)
    b = np.random.uniform(size = size)

    start = clock()
    a + b
    stop = clock()

    return stop - start

if __name__ == '__main__':
  benchmark = ArithmeticBenchmark()
  benchmark.print_result()

  benchmark = NumPyArithmeticBenchmark()
  benchmark.print_result()
