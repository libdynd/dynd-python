import unittest
from dynd import nd, ndt

class TestDType(unittest.TestCase):
    def test_complex_type_realimag(self):
        a = nd.array(1 + 3j)
        self.assertEqual(ndt.complex_float64, nd.type_of(a))
        self.assertEqual(1, nd.as_py(a.real))
        self.assertEqual(3, nd.as_py(a.imag))

        a = nd.array([1 + 2j, 3 + 4j, 5 + 6j])
        self.assertEqual(ndt.type('3 * complex[float64]'), nd.type_of(a))
        self.assertEqual([1, 3, 5], nd.as_py(a.real))
        self.assertEqual([2, 4, 6], nd.as_py(a.imag))

    def test_type_type(self):
        d = ndt.type('type')
        self.assertEqual(str(d), 'type')

        # Creating a dynd array out of a dtype
        # results in it having the dtype 'dtype'
        n = nd.array(d)
        self.assertEqual(nd.type_of(n), d)

        # Python float type converts to float64
        n = nd.array(float)
        self.assertEqual(nd.type_of(n), d)
        self.assertEqual(nd.as_py(n), ndt.float64)

    def test_symbolic_type(self):
        tp = ndt.type('(int, real) -> complex')
        self.assertTrue(ndt.type('Callable').match(tp))
        self.assertEqual(tp.pos_types, [ndt.int32, ndt.float64])
        self.assertEqual(tp.return_type, ndt.complex_float64)
        #tp = ndt.type('MyType')
        #self.assertEqual(tp.type_id, 'typevar')
        #self.assertEqual(tp.name, 'MyType')
        #tp = ndt.type('MyDim * int')
        #self.assertEqual(tp.type_id, 'typevar_dim')
        #self.assertEqual(tp.name, 'MyDim')
        #self.assertEqual(tp.element_type, ndt.int32)
        #tp = ndt.type('... * int')
        #self.assertEqual(tp.type_id, 'ellipsis_dim')
        #self.assertEqual(tp.element_type, ndt.int32)
        #tp = ndt.type('MyEll... * int')
        #self.assertEqual(tp.type_id, 'ellipsis_dim')
        #self.assertEqual(tp.name, 'MyEll')
        #self.assertEqual(tp.element_type, ndt.int32)

    """
    ToDo: Fix this.
    def test_var_dshape(self):
        # Getting the dshape can see into leading var dims
        a = nd.array([[[1], [2,3]]], type='var * var * var * int32')
        self.assertEqual(nd.dshape_of(a), '1 * 2 * var * int32')
    """

if __name__ == '__main__':
    unittest.main(verbosity=2)
