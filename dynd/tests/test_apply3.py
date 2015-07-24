import unittest

from dynd import nd, ndt

class TestApply3(unittest.TestCase):
    def test_annotations(self):
        @nd.functional.apply
        def f(ndt.int32: x) -> ndt.int32:
            return x

        self.assertEqual(ndt.callable(ndt.int32, ndt.int32), f.type)