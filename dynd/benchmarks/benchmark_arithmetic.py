import numpy as np

from dynd import nd, ndt

from benchrun import Benchmark, clock

class ArithemticBenchmark(Benchmark):
  parameters = ('size',)
  size = [100000, 10000000]

  def run(self, size):
    a = nd.uniform(dst_tp = ndt.type('{} * float64'.format(size)))
    b = nd.uniform(dst_tp = ndt.type('{} * float64'.format(size)))

    start = clock()
    a + b
    stop = clock()

    return stop - start

class NumPyArithemticBenchmark(Benchmark):
  parameters = ('size',)
  size = [100000, 10000000]

  def run(self, size):
    a = np.random.uniform(size = size)
    b = np.random.uniform(size = size)

    start = clock()
    a + b
    stop = clock()

    return stop - start

if __name__ == '__main__':
  benchmark = ArithemticBenchmark()
  benchmark.print_result()

  benchmark = NumPyArithemticBenchmark()
  benchmark.print_result()