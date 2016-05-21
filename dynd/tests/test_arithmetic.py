import unittest
from dynd import nd, ndt
import sys

@unittest.skip('Test disabled since translate_exception is not being applied properly')
class TestScalarConstructor(unittest.TestCase):
    def test_arithmetic_exceptions(self):
        # Check that translate_exception is properly
        # used to catch exceptions that arise when using
        # arithmetic operators.
        a = nd.array([1, 2, 3])
        b = nd.array([4, 5])
        self.assertRaises(RuntimeError, a.__add__, b)
        self.assertRaises(RuntimeError, a.__radd__, b)
        self.assertRaises(RuntimeError, a.__sub__, b)
        self.assertRaises(RuntimeError, a.__rsub__, b)
        self.assertRaises(RuntimeError, a.__mul__, b)
        self.assertRaises(RuntimeError, a.__rmul__, b)
        if sys.version_info[0] >= 0:
            self.assertRaises(RuntimeError, a.__truediv__, b)
            self.assertRaises(RuntimeError, a.__rtruediv__, b)
        else:
            self.assertRaises(RuntimeError, a.__div__, b)
            self.assertRaises(RuntimeError, a.__rdiv__, b)
        self.assertRaises(RuntimeError, a.__mod__, b)
        self.assertRaises(RuntimeError, a.__rmod__, b)
        self.assertRaises(RuntimeError, a.__and__, b)
        self.assertRaises(RuntimeError, a.__rand__, b)
        self.assertRaises(RuntimeError, a.__or__, b)
        self.assertRaises(RuntimeError, a.__ror__, b)
        self.assertRaises(RuntimeError, a.__xor__, b)
        self.assertRaises(RuntimeError, a.__rxor__, b)
        self.assertRaises(RuntimeError, a.__lshift__, b)
        self.assertRaises(RuntimeError, a.__rlshift__, b)
        self.assertRaises(RuntimeError, a.__rshift__, b)
        self.assertRaises(RuntimeError, a.__rrshift__, b)
        self.assertRaises(RuntimeError, a.__pow__, b)
        # Check that the optional third argument to pow causes an error
        # if it is used. It should be implemented eventually,
        # but should not fail silently.
        self.assertRaises(ValueError, a.__pow__, a, a)
        self.assertRaises(RuntimeError, a.__rpow__, b)

if __name__ == '__main__':
    unittest.main(verbosity=2)
