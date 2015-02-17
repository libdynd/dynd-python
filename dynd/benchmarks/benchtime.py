from benchrun import clock

class Clock(object):
  def __init__(self, cuda):
    self.cuda = cuda

  def __enter__(self):
    from dynd import cuda

    if self.cuda:
      self.start = cuda.event()
      self.stop = cuda.event()
      self.start.record()
    else:
      self.start = clock()

    return self

  def __exit__(self, type, value, traceback):
    if self.cuda:
      self.stop.record()
      self.stop.synchronize()
    else:
      self.stop = clock()

  def elapsed(self):
    if self.cuda:
      return 1E-3 * self.stop.elapsed_time(self.start)

    return self.stop - self.start

class NumPyClock(object):
  def __enter__(self):
    self.start = clock()
    return self

  def __exit__(self, type, value, traceback):
    self.stop = clock()

  def elapsed(self):
    return self.stop - self.start

class PyCUDAClock(object):
  def __enter__(self):
    from pycuda import driver

    self.start = driver.Event()
    self.stop = driver.Event()

    self.start.record()

    return self

  def __exit__(self, type, value, traceback):
    self.stop.record()
    self.stop.synchronize()

  def elapsed(self):
    return 1E-3 * self.stop.time_since(self.start)