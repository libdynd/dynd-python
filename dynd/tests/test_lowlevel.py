import sys
import unittest
import ctypes
from dynd import nd, ndt, _lowlevel

class TestLowLevel(unittest.TestCase):
    def type_id_of(self, dt):
        assert isinstance(dt, nd.dtype)
        bd = _lowlevel.py_api.get_base_dtype_ptr(dt)
        if bd < _lowlevel.BUILTIN_TYPE_ID_COUNT:
            return bd
        else:
            bdm = _lowlevel.BaseDTypeMembers.from_address(
                            _lowlevel.api.get_base_dtype_members(bd))
            return bdm.type_id

    def test_type_id(self):
        # Numeric type id
        self.assertEqual(self.type_id_of(ndt.bool),
                        _lowlevel.BOOL_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.int8),
                        _lowlevel.INT8_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.int16),
                        _lowlevel.INT16_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.int32),
                        _lowlevel.INT32_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.int64),
                        _lowlevel.INT64_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.uint8),
                        _lowlevel.UINT8_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.uint16),
                        _lowlevel.UINT16_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.uint32),
                        _lowlevel.UINT32_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.uint64),
                        _lowlevel.UINT64_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.float32),
                        _lowlevel.FLOAT32_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.float64),
                        _lowlevel.FLOAT64_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.cfloat32),
                        _lowlevel.COMPLEX_FLOAT32_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.cfloat64),
                        _lowlevel.COMPLEX_FLOAT64_TYPE_ID)
        # String/bytes
        self.assertEqual(self.type_id_of(ndt.string),
                        _lowlevel.STRING_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.make_fixedstring_dtype(16)),
                        _lowlevel.FIXEDSTRING_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.bytes),
                        _lowlevel.BYTES_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.make_fixedbytes_dtype(16)),
                        _lowlevel.FIXEDBYTES_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.json),
                        _lowlevel.JSON_TYPE_ID)
        # Date
        self.assertEqual(self.type_id_of(ndt.date),
                        _lowlevel.DATE_TYPE_ID)
        # Property
        self.assertEqual(self.type_id_of(ndt.date(2000, 1, 1).year.dtype),
                        _lowlevel.PROPERTY_TYPE_ID)
        # Categorical
        self.assertEqual(self.type_id_of(ndt.make_categorical_dtype([1, 2, 3])),
                        _lowlevel.CATEGORICAL_TYPE_ID)
        # Struct
        self.assertEqual(self.type_id_of(ndt.make_struct_dtype(
                                    [ndt.int32, ndt.int32], ['x', 'y'])),
                        _lowlevel.STRUCT_TYPE_ID)
        self.assertEqual(self.type_id_of(nd.dtype('{x : int32; y : int32}')),
                        _lowlevel.FIXEDSTRUCT_TYPE_ID)
        # Convert/byteswap/view
        self.assertEqual(self.type_id_of(ndt.make_convert_dtype(
                                    ndt.int32, ndt.int8)),
                        _lowlevel.CONVERT_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.make_byteswap_dtype(ndt.int32)),
                        _lowlevel.BYTESWAP_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.make_view_dtype(
                                    ndt.int32, ndt.uint32)),
                        _lowlevel.VIEW_TYPE_ID)
        # Uniform arrays
        self.assertEqual(self.type_id_of(nd.dtype('3, int32')),
                        _lowlevel.FIXED_DIM_TYPE_ID)
        self.assertEqual(self.type_id_of(nd.dtype('M, int32')),
                        _lowlevel.STRIDED_DIM_TYPE_ID)
        self.assertEqual(self.type_id_of(nd.dtype('var, int32')),
                        _lowlevel.VAR_DIM_TYPE_ID)
        # GroupBy
        self.assertEqual(self.type_id_of(nd.groupby([1, 2], ['a', 'a']).dtype),
                        _lowlevel.GROUPBY_TYPE_ID)
        # DType
        self.assertEqual(self.type_id_of(nd.dtype('dtype')),
                        _lowlevel.DTYPE_TYPE_ID)

    def test_ndobject_from_ptr(self):
        a = (ctypes.c_int32 * 3)()
        a[0] = 3
        a[1] = 6
        a[2] = 9
        # Readwrite version
        b = _lowlevel.py_api.ndobject_from_ptr(nd.dtype('3, int32'), ctypes.addressof(a),
                        a, 'readwrite')
        self.assertEqual(_lowlevel.data_address_of(b), ctypes.addressof(a))
        self.assertEqual(b.dshape, '3, int32')
        self.assertEqual(nd.as_py(b), [3, 6, 9])
        b[1] = 10
        self.assertEqual(a[1], 10)
        # Readonly version
        b = _lowlevel.py_api.ndobject_from_ptr(nd.dtype('3, int32'), ctypes.addressof(a),
                        a, 'readonly')
        self.assertEqual(nd.as_py(b), [3, 10, 9])
        def assign_to(b):
            b[1] = 100
        self.assertRaises(RuntimeError, assign_to, b)

    def test_ndobject_from_ptr_error(self):
        # Should raise an exception if the dtype has metadata
        a = (ctypes.c_int32 * 4)()
        self.assertRaises(RuntimeError, _lowlevel.py_api.ndobject_from_ptr,
                        nd.dtype('M, int32'), ctypes.addressof(a),
                        a, 'readwrite')

if __name__ == '__main__':
    unittest.main()
