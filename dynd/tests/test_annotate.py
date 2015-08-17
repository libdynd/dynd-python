import sys
if sys.version_info >= (2, 7):
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
        if sys.version_info >= (3, 0):
            exec('''def f3(): pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        with self.assertRaises(TypeError):
            @annotate(x = float)
            def f():
                pass

    def test_nullary_with_return(self):
        @annotate(int)
        def f():
            pass

        self.assertEqual({'return': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3() -> int: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(float)
        def f():
            pass

        self.assertEqual({'return': float}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3() -> float: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

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
            pass

        self.assertFalse(f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x): pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(x = int)
        def f(x):
            pass

        self.assertEqual({'x': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: int): pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(x = float)
        def f(x):
            pass

        self.assertEqual({'x': float}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: float): pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        with self.assertRaises(TypeError):
            @annotate(x = int, y = str)
            def f(x):
                pass

    def test_unary_with_return(self):
        @annotate(int)
        def f(x):
            pass

        self.assertEqual({'return': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x) -> int: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(float)
        def f(x):
            pass

        self.assertEqual({'return': float}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x) -> float: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(int, int)
        def f(x):
            pass

        self.assertEqual({'return': int, 'x': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: int) -> int: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(int, x = int)
        def f(x):
            pass

        self.assertEqual({'return': int, 'x': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: int) -> int: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(int, float)
        def f(x):
            pass

        self.assertEqual({'return': int, 'x': float}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: float) -> int: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(int, x = float)
        def f(x):
            pass

        self.assertEqual({'return': int, 'x': float}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: float) -> int: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(float, int)
        def f(x):
            pass

        self.assertEqual({'return': float, 'x': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: int) -> float: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(float, x = int)
        def f(x):
            pass

        self.assertEqual({'return': float, 'x': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: int) -> float: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        with self.assertRaises(TypeError):
            @annotate(float, int, str)
            def f(x):
                pass

        with self.assertRaises(TypeError):
            @annotate(float, x = int, y = str)
            def f(x):
                pass

        with self.assertRaises(TypeError):
            @annotate(float, int, x = str)
            def f(x):
                pass

    def test_binary(self):
        @annotate()
        def f(x, y):
            pass

        self.assertFalse(f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x, y): pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(x = int)
        def f(x, y):
            pass

        self.assertEqual({'x': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: int, y): pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(x = float)
        def f(x, y):
            pass

        self.assertEqual({'x': float}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: float, y): pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(x = int, y = int)
        def f(x, y):
            pass

        self.assertEqual({'x': int, 'y': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: int, y: int): pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(x = float, y = int)
        def f(x, y):
            pass

        self.assertEqual({'x': float, 'y': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: float, y: int): pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        with self.assertRaises(TypeError):
            @annotate(x = int, y = int, z = str)
            def f(x, y):
                pass

    def test_binary_with_return(self):
        @annotate(int)
        def f(x, y):
            pass

        self.assertEqual({'return': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x, y) -> int: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(float)
        def f(x, y):
            pass

        self.assertEqual({'return': float}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x, y) -> float: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(int, int)
        def f(x, y):
            pass

        self.assertEqual({'return': int, 'x': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: int, y) -> int: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(int, x = int)
        def f(x, y):
            pass

        self.assertEqual({'return': int, 'x': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: int, y) -> int: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(int, float)
        def f(x, y):
            pass

        self.assertEqual({'return': int, 'x': float}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: float, y) -> int: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(int, x = float)
        def f(x, y):
            pass

        self.assertEqual({'return': int, 'x': float}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: float, y) -> int: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(float, int)
        def f(x, y):
            pass

        self.assertEqual({'return': float, 'x': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: int, y) -> float: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(float, x = int)
        def f(x, y):
            pass

        self.assertEqual({'return': float, 'x': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: int, y) -> float: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(int, int, int)
        def f(x, y):
            pass

        self.assertEqual({'return': int, 'x': int, 'y': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: int, y: int) -> int: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(int, x = int, y = int)
        def f(x, y):
            pass

        self.assertEqual({'return': int, 'x': int, 'y': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: int, y: int) -> int: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(int, float, int)
        def f(x, y):
            pass

        self.assertEqual({'return': int, 'x': float, 'y': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: float, y: int) -> int: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(int, x = float, y = int)
        def f(x, y):
            pass

        self.assertEqual({'return': int, 'x': float, 'y': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: float, y: int) -> int: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(float, int, int)
        def f(x, y):
            pass

        self.assertEqual({'return': float, 'x': int, 'y': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: int, y: int) -> float: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        @annotate(float, x = int, y = int)
        def f(x, y):
            pass

        self.assertEqual({'return': float, 'x': int, 'y': int}, f.__annotations__)
        if sys.version_info >= (3, 0):
            exec('''def f3(x: int, y: int) -> float: pass''', globals())
            self.assertEqual(f3.__annotations__, f.__annotations__)

        with self.assertRaises(TypeError):
            @annotate(float, int, int, str)
            def f(x, y):
                pass

        with self.assertRaises(TypeError):
            @annotate(float, x = int, y = int, z = str)
            def f(x, y):
                pass

        with self.assertRaises(TypeError):
            @annotate(float, int, x = str)
            def f(x, y):
                pass

        with self.assertRaises(TypeError):
            @annotate(float, int, float, y = str)
            def f(x, y):
                pass

if __name__ == '__main__':
    unittest.main()
