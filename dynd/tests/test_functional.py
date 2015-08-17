import unittest

from dynd import annotate, nd, ndt

class TestApply(unittest.TestCase):
    def test_object(self):
        @nd.functional.apply(jit = False)
        def f(x):
            return x

        self.assertEqual(f.type, ndt.callable(ndt.scalar, ndt.scalar))

        @nd.functional.apply(jit = False)
        @annotate(ndt.int32)
        def f(x):
            return x

        self.assertEqual(f.type, ndt.callable(ndt.int32, ndt.scalar))
        self.assertEqual(0, f(0))
        self.assertEqual(1, f(1))
        self.assertEqual(0.0, f(0.0))
        self.assertEqual(1.0, f(1.0))

        @nd.functional.apply(jit = False)
        @annotate(ndt.int32, ndt.int32)
        def f(x):
            return x

        self.assertEqual(f.type, ndt.callable(ndt.int32, ndt.int32))
        self.assertEqual(0, f(0))
        self.assertEqual(1, f(1))

    def test_numba(self):
        try:
            import numba
        except ImportError as error:
            raise unittest.SkipTest(error.message)

        @nd.functional.apply(jit = True)
        def f(x):
            return x

        self.assertEqual(f.type, ndt.callable(ndt.scalar, ndt.scalar))
        self.assertEqual(0, f(0))

        @nd.functional.apply(jit = True)
        def f(x):
            return x

        self.assertEqual(f.type, ndt.callable(ndt.scalar, ndt.scalar))
        self.assertEqual(0, f(0))

if __name__ == '__main__':
    unittest.main()
