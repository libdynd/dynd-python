import sys
if sys.version_info >= (2, 7)
    import unittest
else:
    import unittest2 as unittest

from dynd import annotate

class TestAnnotate(unittest.TestCase):
    def test_nullary(self):
        @annotate()
        def f():
            pass

        self.assertFalse(f.__annotations__)

        with self.assertRaises(TypeError):
            @annotate(x = float)
            def f():
                pass

    def test_nullary_with_return(self):
        @annotate(int)
        def f():
            pass

        self.assertEqual({'return': int}, f.__annotations__)

        @annotate(float)
        def f():
            pass

        self.assertEqual({'return': float}, f.__annotations__)

        with self.assertRaises(TypeError):
            @annotate(int, float)
            def f():
                pass

        with self.assertRaises(TypeError):
            @annotate(int, float, float)
            def f():
                pass

        with self.assertRaises(TypeError):
            @annotate(int, x = float)
            def f():
                pass

    def test_unary(self):
        @annotate()
        def f(x):
            return x

        self.assertFalse(f.__annotations__)

        @annotate(x = int)
        def f(x):
            return x

        self.assertEqual({'x': int}, f.__annotations__)

        @annotate(x = float)
        def f(x):
            return x

        self.assertEqual({'x': float}, f.__annotations__)

        with self.assertRaises(TypeError):
            @annotate(x = int, y = str)
            def f(x):
                return x

    def test_unary_with_return(self):
        @annotate(int)
        def f(x):
            return x

        self.assertEqual({'return': int}, f.__annotations__)

        @annotate(float)
        def f(x):
            return x

        self.assertEqual({'return': float}, f.__annotations__)

        @annotate(int, int)
        def f(x):
            return x

        self.assertEqual({'return': int, 'x': int}, f.__annotations__)

        @annotate(int, x = int)
        def f(x):
            return x

        self.assertEqual({'return': int, 'x': int}, f.__annotations__)

        @annotate(int, float)
        def f(x):
            return x

        self.assertEqual({'return': int, 'x': float}, f.__annotations__)

        @annotate(int, x = float)
        def f(x):
            return x

        self.assertEqual({'return': int, 'x': float}, f.__annotations__)

        @annotate(float, int)
        def f(x):
            return x

        self.assertEqual({'return': float, 'x': int}, f.__annotations__)

        @annotate(float, x = int)
        def f(x):
            return x

        self.assertEqual({'return': float, 'x': int}, f.__annotations__)

        with self.assertRaises(TypeError):
            @annotate(float, int, str)
            def f(x):
                return x

        with self.assertRaises(TypeError):
            @annotate(float, x = int, y = str)
            def f(x):
                return x

        with self.assertRaises(TypeError):
            @annotate(float, int, x = str)
            def f(x):
                return x

"""
    def test_python3(self):
        @annotate(int, x = int)
        def f(x):
            return x

        def g(x: int) -> int:
            return x

        self.assertEqual(g.__annotations__, f.__annotations__)
"""

if __name__ == '__main__':
    unittest.main()
