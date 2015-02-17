from benchrun import clock

class Timer(object):
  def __enter__(self):
    self.start = clock()

    return self

  def __exit__(self, type, value, traceback):
      self.stop = clock()

  def elapsed_time(self):
    return self.stop - self.start

class CUDATimer(object):
  def __enter__(self):
    from dynd import cuda

    self.start = cuda.event()
    self.stop = cuda.event()
    self.start.record()

    return self

  def __exit__(self, type, value, traceback):
    self.stop.record()
    self.stop.synchronize()

  def elapsed_time(self):
    return 1E-3 * self.stop.elapsed_time(self.start)