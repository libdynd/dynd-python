import sys
import unittest
from dynd import nd, ndt

class TestTypePatternMatch(unittest.TestCase):
    def test_simple(self):
        self.assertTrue(ndt.int32.matches(ndt.int32))
        self.assertTrue(ndt.int16.matches('T'))
        self.assertTrue(ndt.int16.matches('... * T'))
        self.assertTrue(ndt.int16.matches('A... * T'))
        self.assertTrue(ndt.type('strided * var * int').matches('M * A... * N * T'))
        self.assertFalse(ndt.type('strided * int').matches('M * A... * N * T'))

    def test_tuple(self):
        pat = ndt.type('(T, ?T, 3 * T, A... * S)')
        self.assertTrue(ndt.type('(int, ?int, 3 * int, real)').matches(pat))
        self.assertTrue(ndt.type('(string, ?string, 3 * string, 10 * complex)').matches(pat))
        self.assertFalse(ndt.type('(string, ?int, 3 * string, 10 * complex)').matches(pat))
        self.assertFalse(ndt.type('(string, string, 3 * string, 10 * complex)').matches(pat))
        self.assertFalse(ndt.type('(string, ?string, 4 * string, 10 * complex)').matches(pat))
