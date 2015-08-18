import sys
if sys.version_info >= (2, 7):
    import unittest
else:
    import unittest2 as unittest

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
            raise unittest.SkipTest(error)

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

class TestElwise(unittest.TestCase):
    def test_unary(self):
        @nd.functional.elwise
        @annotate(ndt.int32)
        def f(x):
            return 2 * x

#        self.assertEqual(nd.array([2, 4, 6]), f([1, 2, 3]))

class TestReduction(unittest.TestCase):
    def test_unary(self):
        @nd.functional.reduction
        @annotate(ndt.int32)
        def f(x, y):
            return max(x, y)

        self.assertEqual(3, f([1, 2, 3]))
        self.assertEqual(6, f([[1, 2, 3], [4, 5, 6]]))

"""
def multigen(func):
    return lambda x: x

class TestMultidispatch(unittest.TestCase):
    def test_unary(self):
        @nd.functional.multidispatch()
        def callables():
            yield 5

        print callables(3)
"""

if __name__ == '__main__':
    unittest.main()
