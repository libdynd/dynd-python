import sys
import unittest
import ctypes
from dynd import nd, ndt

"""
class TestLowLevel(unittest.TestCase):
    def type_id_of(self, dt):
        assert isinstance(dt, ndt.type)
        bd = _lowlevel.get_base_type_ptr(dt)
        if bd < _lowlevel.type_id.BUILTIN_TYPE_ID_COUNT:
            return bd
        else:
            bdm = _lowlevel.BaseDTypeMembers.from_address(
                            _lowlevel.get_base_type_members(bd))
            return bdm.type_id

    def test_type_id(self):
        # Numeric type id
        self.assertEqual(self.type_id_of(ndt.bool),
                        _lowlevel.type_id.BOOL)
        self.assertEqual(self.type_id_of(ndt.int8),
                        _lowlevel.type_id.INT8)
        self.assertEqual(self.type_id_of(ndt.int16),
                        _lowlevel.type_id.INT16)
        self.assertEqual(self.type_id_of(ndt.int32),
                        _lowlevel.type_id.INT32)
        self.assertEqual(self.type_id_of(ndt.int64),
                        _lowlevel.type_id.INT64)
        self.assertEqual(self.type_id_of(ndt.uint8),
                        _lowlevel.type_id.UINT8)
        self.assertEqual(self.type_id_of(ndt.uint16),
                        _lowlevel.type_id.UINT16)
        self.assertEqual(self.type_id_of(ndt.uint32),
                        _lowlevel.type_id.UINT32)
        self.assertEqual(self.type_id_of(ndt.uint64),
                        _lowlevel.type_id.UINT64)
        self.assertEqual(self.type_id_of(ndt.float32),
                        _lowlevel.type_id.FLOAT32)
        self.assertEqual(self.type_id_of(ndt.float64),
                        _lowlevel.type_id.FLOAT64)
        self.assertEqual(self.type_id_of(ndt.complex_float32),
                        _lowlevel.type_id.COMPLEX_FLOAT32)
        self.assertEqual(self.type_id_of(ndt.complex_float64),
                        _lowlevel.type_id.COMPLEX_FLOAT64)
        # String/bytes
        self.assertEqual(self.type_id_of(ndt.string),
                        _lowlevel.type_id.STRING)
        self.assertEqual(self.type_id_of(ndt.make_fixed_string(16)),
                        _lowlevel.type_id.FIXED_STRING)
        self.assertEqual(self.type_id_of(ndt.bytes),
                        _lowlevel.type_id.BYTES)
        self.assertEqual(self.type_id_of(ndt.make_fixed_bytes(16)),
                        _lowlevel.type_id.FIXED_BYTES)
        self.assertEqual(self.type_id_of(ndt.json),
                        _lowlevel.type_id.JSON)
        # Date
        self.assertEqual(self.type_id_of(ndt.date),
                        _lowlevel.type_id.DATE)
        self.assertEqual(self.type_id_of(ndt.time),
                        _lowlevel.type_id.TIME)
        self.assertEqual(self.type_id_of(ndt.datetime),
                        _lowlevel.type_id.DATETIME)
        # Property
#        self.assertEqual(self.type_id_of(nd.type_of(ndt.date(2000, 1, 1).year)),
 #                       _lowlevel.type_id.PROPERTY)
        # Categorical
        self.assertEqual(self.type_id_of(ndt.make_categorical([1, 2, 3])),
                        _lowlevel.type_id.CATEGORICAL)
        # Struct
        self.assertEqual(self.type_id_of(ndt.make_struct(
                                    [ndt.int32, ndt.int32], ['x', 'y'])),
                        _lowlevel.type_id.STRUCT)
        self.assertEqual(self.type_id_of(ndt.type('{x : int32, y : int32}')),
                        _lowlevel.type_id.STRUCT)
        # Tuple
        self.assertEqual(self.type_id_of(ndt.type('(int32, int32)')),
                        _lowlevel.type_id.TUPLE)
        # NDArrayArg
        self.assertEqual(self.type_id_of(ndt.type('ndarrayarg')),
                         _lowlevel.type_id.NDARRAYARG)
        # Adapt/convert/byteswap/view
        self.assertEqual(
            self.type_id_of(ndt.type('adapt[(date) -> int, "days since 1970-01-01"]')),
            _lowlevel.type_id.ADAPT)
        self.assertEqual(self.type_id_of(ndt.make_convert(
                                    ndt.int32, ndt.int8)),
                        _lowlevel.type_id.CONVERT)
        self.assertEqual(self.type_id_of(ndt.make_byteswap(ndt.int32)),
                        _lowlevel.type_id.BYTESWAP)
        self.assertEqual(self.type_id_of(ndt.make_view(
                                    ndt.int32, ndt.uint32)),
                        _lowlevel.type_id.VIEW)
        # CUDA types
#        if ndt.cuda_support:
 #           self.assertEqual(self.type_id_of(ndt.type('cuda_device[int32]')),
  #                           _lowlevel.type_id.CUDA_DEVICE)
   #         self.assertEqual(self.type_id_of(ndt.type('cuda_host[int32]')),
    #                         _lowlevel.type_id.CUDA_HOST)
        # Uniform arrays
        self.assertEqual(self.type_id_of(ndt.type('fixed[3] * int32')),
                        _lowlevel.type_id.FIXED_DIM)
        self.assertEqual(self.type_id_of(ndt.type('Fixed * int32')),
                        _lowlevel.type_id.FIXED_DIM)
        self.assertEqual(self.type_id_of(ndt.type('var * int32')),
                        _lowlevel.type_id.VAR_DIM)
        # GroupBy
        self.assertEqual(self.type_id_of(nd.type_of(nd.groupby([1, 2],
                                                               ['a', 'a']))),
                        _lowlevel.type_id.GROUPBY)
        # Type
        self.assertEqual(self.type_id_of(ndt.type('type')),
                        _lowlevel.type_id.TYPE)
"""

if __name__ == '__main__':
    unittest.main(verbosity=2)
