import sys
import unittest
import ctypes
from dynd import nd, ndt, lowlevel

class TestLowLevel(unittest.TestCase):
    def type_id_of(self, dt):
        assert isinstance(dt, nd.dtype)
        bd = lowlevel.py_api.get_base_dtype_ptr(dt)
        if bd < lowlevel.BUILTIN_TYPE_ID_COUNT:
            return bd
        else:
            bdm = lowlevel.BaseDTypeMembers.from_address(
                            lowlevel.api.get_base_dtype_members(bd))
            return bdm.type_id

    def test_type_id(self):
        # Numeric type id
        self.assertEqual(self.type_id_of(ndt.bool),
                        lowlevel.BOOL_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.int8),
                        lowlevel.INT8_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.int16),
                        lowlevel.INT16_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.int32),
                        lowlevel.INT32_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.int64),
                        lowlevel.INT64_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.uint8),
                        lowlevel.UINT8_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.uint16),
                        lowlevel.UINT16_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.uint32),
                        lowlevel.UINT32_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.uint64),
                        lowlevel.UINT64_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.float32),
                        lowlevel.FLOAT32_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.float64),
                        lowlevel.FLOAT64_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.cfloat32),
                        lowlevel.COMPLEX_FLOAT32_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.cfloat64),
                        lowlevel.COMPLEX_FLOAT64_TYPE_ID)
        # String/bytes
        self.assertEqual(self.type_id_of(ndt.string),
                        lowlevel.STRING_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.make_fixedstring_dtype(16)),
                        lowlevel.FIXEDSTRING_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.bytes),
                        lowlevel.BYTES_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.make_fixedbytes_dtype(16)),
                        lowlevel.FIXEDBYTES_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.json),
                        lowlevel.JSON_TYPE_ID)
        # Date
        self.assertEqual(self.type_id_of(ndt.date),
                        lowlevel.DATE_TYPE_ID)
        # Property
        self.assertEqual(self.type_id_of(ndt.date(2000, 1, 1).year.dtype),
                        lowlevel.PROPERTY_TYPE_ID)
        # Categorical
        self.assertEqual(self.type_id_of(ndt.make_categorical_dtype([1, 2, 3])),
                        lowlevel.CATEGORICAL_TYPE_ID)
        # Struct
        self.assertEqual(self.type_id_of(ndt.make_struct_dtype(
                                    [ndt.int32, ndt.int32], ['x', 'y'])),
                        lowlevel.STRUCT_TYPE_ID)
        self.assertEqual(self.type_id_of(nd.dtype('{x : int32; y : int32}')),
                        lowlevel.FIXEDSTRUCT_TYPE_ID)
        # Convert/byteswap/view
        self.assertEqual(self.type_id_of(ndt.make_convert_dtype(
                                    ndt.int32, ndt.int8)),
                        lowlevel.CONVERT_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.make_byteswap_dtype(ndt.int32)),
                        lowlevel.BYTESWAP_TYPE_ID)
        self.assertEqual(self.type_id_of(ndt.make_view_dtype(
                                    ndt.int32, ndt.uint32)),
                        lowlevel.VIEW_TYPE_ID)
        # Uniform arrays
        self.assertEqual(self.type_id_of(nd.dtype('3, int32')),
                        lowlevel.FIXED_DIM_TYPE_ID)
        self.assertEqual(self.type_id_of(nd.dtype('M, int32')),
                        lowlevel.STRIDED_DIM_TYPE_ID)
        self.assertEqual(self.type_id_of(nd.dtype('VarDim, int32')),
                        lowlevel.VAR_DIM_TYPE_ID)
        # GroupBy
        self.assertEqual(self.type_id_of(nd.groupby([1, 2], ['a', 'a']).dtype),
                        lowlevel.GROUPBY_TYPE_ID)
        # DType
        self.assertEqual(self.type_id_of(nd.dtype('dtype')),
                        lowlevel.DTYPE_TYPE_ID)

if __name__ == '__main__':
    unittest.main()
