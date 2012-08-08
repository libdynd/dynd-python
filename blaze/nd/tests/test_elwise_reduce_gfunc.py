import sys
import unittest
from blaze import nd

class TestElwiseReduceGFunc(unittest.TestCase):

    def test_sum(self):
        # Tests an elementwise reduce gfunc with identity
        x = nd.sum([1,2,3,4,5]).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), 15)

        x = nd.sum([1,2,3,4,5], associate='left').vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), 15)

        x = nd.sum([1,2,3,4,5], associate='right').vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), 15)

        x = nd.sum([1,2,3,4,5], keepdims=True).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), [15])

        x = nd.sum([[1,2,3],[4,5,6]]).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), 21)

        x = nd.sum([[1,2,3],[4,5,6]], axis=0).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), [5,7,9])

        x = nd.sum([[1,2,3],[4,5,6]], axis=1).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), [6,15])

        x = nd.sum([[1,2,3],[4,5,6]], axis=0, keepdims=True).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), [[5,7,9]])

        x = nd.sum([[1,2,3],[4,5,6]], axis=1, keepdims=True).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), [[6],[15]])

    def test_max(self):
        # Tests an elementwise reduce gfunc without an identity
        x = nd.max([1,2,8,4,5]).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), 8)

        x = nd.max([1,2,8,4,5], associate='left').vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), 8)

        x = nd.max([1,2,8,4,5], associate='right').vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), 8)

        x = nd.max([1,2,8,4,5], keepdims=True).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), [8])

        x = nd.max([[1,8,3],[4,5,6]]).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), 8)

        x = nd.max([[1,8,3],[4,5,6]], axis=0).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), [4,8,6])

        x = nd.max([[1,8,3],[4,5,6]], axis=1).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), [8,6])

        x = nd.max([[1,8,3],[4,5,6]], axis=0, keepdims=True).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), [[4,8,6]])

        x = nd.max([[1,8,3],[4,5,6]], axis=1, keepdims=True).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), [[8],[6]])

    def test_repeated_creation(self):
        """Tests that repeated creation/destruction of the gfunc is ok"""
        for i in range(10):
            gf = nd.gfunc.elwise_reduce('test_repeated_creation_gf')
            gf.add_kernel(nd.elwise_kernels.add_int32, commutative=True,
                                associative=True)
            self.assertEqual(gf([3,4]).as_py(), 7)

if __name__ == '__main__':
    unittest.main()


