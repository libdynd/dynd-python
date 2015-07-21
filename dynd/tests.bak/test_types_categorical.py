import sys
import unittest
from dynd import nd, ndt

"""
Todo: Fix this.

class TestDType(unittest.TestCase):
    def test_make_categorical(self):
        # Create categorical type with 256 values
        tp = ndt.make_categorical(nd.range(0, 512, 2))
        self.assertEqual(tp.type_id, 'categorical')
        self.assertEqual(tp.storage_type, ndt.uint8)
        self.assertEqual(tp.category_type, ndt.int32)
        # Create categorical type with 256 < x < 65536 values
        tp = ndt.make_categorical(nd.range(40000, dtype=ndt.float32))
        self.assertEqual(tp.type_id, 'categorical')
        self.assertEqual(tp.storage_type, ndt.uint16)
        self.assertEqual(tp.category_type, ndt.float32)
        # Create categorical type with > 65536 values
        tp = ndt.make_categorical(nd.range(70000, dtype=ndt.int128))
        self.assertEqual(tp.type_id, 'categorical')
        self.assertEqual(tp.storage_type, ndt.uint32)
        self.assertEqual(tp.category_type, ndt.int128)

    def test_factor_categorical(self):
        a = nd.array(["2012-05-10T02:29:42"] * 100, "datetime")
        dt1 = ndt.factor_categorical(a.date)
        #print (dt1)
#        self.assertEqual(nd.as_py(dt1.categories.ucast(ndt.string)),
 #                       ['2012-05-10'])

    def test_factor_fixedstring(self):
        adata = [('M', 13), ('F', 17), ('F', 34), ('M', 19),
                 ('M', 13), ('F', 34), ('F', 22)]
        a = nd.array(adata, dtype='{gender: fixed_string[1], age: int32}')
        catdt = ndt.factor_categorical(a)
        b = a.ucast(catdt)
        x = repr(b)
        self.assertTrue('["M", 13]' in x)

    def test_rainbow_example(self):
        rainbow_vals = ['red', 'orange', 'yellow',
                        'green', 'blue', 'indigo', 'violet']
        color_vals = ['red', 'red', 'violet', 'blue',
                      'yellow', 'yellow', 'red', 'indigo']
        color_vals_int = [rainbow_vals.index(x) for x in color_vals]
        # Create the type
        rainbow = ndt.make_categorical(rainbow_vals)
        # Make sure it looks the way we expect
        self.assertEqual(rainbow.type_id, 'categorical')
        self.assertEqual(rainbow.data_size, 1)
        self.assertEqual(rainbow.data_alignment, 1)
        self.assertEqual(rainbow.storage_type, ndt.uint8)
        self.assertEqual(rainbow.category_type, ndt.string)
        self.assertEqual(nd.as_py(rainbow.categories), rainbow_vals)
        # Create an array of the type
        colors = nd.array(color_vals, dtype=rainbow)
        # Make sure it is convertible back to strings/pyobject/int
        self.assertEqual(nd.as_py(colors), color_vals)
        self.assertEqual(nd.as_py(colors.ints), color_vals_int)
"""

if __name__ == '__main__':
    unittest.main()
