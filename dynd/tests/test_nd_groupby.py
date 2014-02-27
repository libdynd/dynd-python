import sys
import unittest
from dynd import nd, ndt

class TestGroupBy(unittest.TestCase):
    def test_immutable(self):
        a = nd.array([
                ('x', 0),
                ('y', 1),
                ('x', 2),
                ('x', 3),
                ('y', 4)],
                dtype='{A: string, B: int32}').eval_immutable()
        gb = nd.groupby(a, nd.fields(a, 'A'))
        self.assertEqual(nd.as_py(gb.groups), [{'A': 'x'}, {'A': 'y'}])
        self.assertEqual(nd.as_py(gb), [
                [{'A': 'x', 'B': 0},
                 {'A': 'x', 'B': 2},
                 {'A': 'x', 'B': 3}],
                [{'A': 'y', 'B': 1},
                 {'A': 'y', 'B': 4}]])

    def test_grouped_slices(self):
        a = nd.asarray([[1, 2, 3], [1, 4, 5]])
        gb = nd.groupby(a[:, 1:], a[:, 0])
        self.assertEqual(nd.as_py(gb.groups), [1])
        self.assertEqual(nd.as_py(gb), [[[2, 3], [4, 5]]])

        a = nd.asarray([[1, 2, 3], [3, 1, 7], [1, 4, 5], [2, 6, 7], [3, 2, 5]])
        gb = nd.groupby(a[:, 1:], a[:, 0])
        self.assertEqual(nd.as_py(gb.groups), [1, 2, 3])
        self.assertEqual(nd.as_py(gb), [[[2, 3], [4, 5]],
                                        [[6, 7]],
                                        [[1, 7], [2, 5]]])

if __name__ == '__main__':
    unittest.main()
