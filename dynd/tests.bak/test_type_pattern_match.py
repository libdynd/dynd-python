import sys
import unittest
from dynd import nd, ndt

class TestTypePatternMatch(unittest.TestCase):
    def test_simple(self):
        self.assertTrue(ndt.int32.match(ndt.int32))
        self.assertTrue(ndt.type('T').match(ndt.int16))
        self.assertTrue(ndt.type('... * T').match(ndt.int16))
        self.assertTrue(ndt.type('A... * T').match(ndt.int16))
        self.assertTrue(ndt.type('M * A... * N * T').match(ndt.type('Fixed * var * int')))
        self.assertFalse(ndt.type('M * A... * N * T').match(ndt.type('Fixed * int')))

    def test_tuple(self):
        pat = ndt.type('(T, ?T, 3 * T, A... * S)')
        self.assertTrue(pat.match(ndt.type('(int, ?int, 3 * int, real)')))
        self.assertTrue(pat.match(ndt.type('(string, ?string, 3 * string, 10 * complex)')))
        self.assertFalse(pat.match(ndt.type('(string, ?int, 3 * string, 10 * complex)')))
        self.assertFalse(pat.match(ndt.type('(string, string, 3 * string, 10 * complex)')))
        self.assertFalse(pat.match(ndt.type('(string, ?string, 4 * string, 10 * complex)')))

if __name__ == '__main__':
    unittest.main()
