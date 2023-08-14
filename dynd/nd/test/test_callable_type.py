import unittest
# import nd to ensure that the callable type is registered at all.
from dynd import nd, ndt

class TestCallableType(unittest.TestCase):

    def test_callable(self):
        tp = ndt.callable(ndt.void, ndt.int32, ndt.float64, x = ndt.complex128)

    def test_callable_type(self):
        tp = ndt.callable(ndt.int32, ndt.float64)
