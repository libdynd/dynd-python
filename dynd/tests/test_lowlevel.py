import sys
import unittest
import ctypes
from dynd import nd, ndt, _lowlevel

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
        self.assertEqual(self.type_id_of(ndt.make_fixedstring(16)),
                        _lowlevel.type_id.FIXEDSTRING)
        self.assertEqual(self.type_id_of(ndt.bytes),
                        _lowlevel.type_id.BYTES)
        self.assertEqual(self.type_id_of(ndt.make_fixedbytes(16)),
                        _lowlevel.type_id.FIXEDBYTES)
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
        self.assertEqual(self.type_id_of(nd.type_of(ndt.date(2000, 1, 1).year)),
                        _lowlevel.type_id.PROPERTY)
        # Categorical
        self.assertEqual(self.type_id_of(ndt.make_categorical([1, 2, 3])),
                        _lowlevel.type_id.CATEGORICAL)
        # Struct
        self.assertEqual(self.type_id_of(ndt.make_struct(
                                    [ndt.int32, ndt.int32], ['x', 'y'])),
                        _lowlevel.type_id.STRUCT)
        self.assertEqual(self.type_id_of(ndt.type('{x : int32, y : int32}')),
                        _lowlevel.type_id.STRUCT)
        self.assertEqual(self.type_id_of(ndt.type('c{x : int32, y : int32}')),
                        _lowlevel.type_id.CSTRUCT)
        # Tuple
        self.assertEqual(self.type_id_of(ndt.type('(int32, int32)')),
                        _lowlevel.type_id.TUPLE)
        self.assertEqual(self.type_id_of(ndt.type('c(int32, int32)')),
                        _lowlevel.type_id.CTUPLE)
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
        if ndt.cuda_support:
            self.assertEqual(self.type_id_of(ndt.type('cuda_device[int32]')),
                             _lowlevel.type_id.CUDA_DEVICE)
            self.assertEqual(self.type_id_of(ndt.type('cuda_host[int32]')),
                             _lowlevel.type_id.CUDA_HOST)
        # Uniform arrays
        self.assertEqual(self.type_id_of(ndt.type('cfixed[3] * int32')),
                        _lowlevel.type_id.CFIXED_DIM)
        self.assertEqual(self.type_id_of(ndt.type('fixed[3] * int32')),
                        _lowlevel.type_id.FIXED_DIM)
        self.assertEqual(self.type_id_of(ndt.type('strided * int32')),
                        _lowlevel.type_id.STRIDED_DIM)
        self.assertEqual(self.type_id_of(ndt.type('var * int32')),
                        _lowlevel.type_id.VAR_DIM)
        # GroupBy
        self.assertEqual(self.type_id_of(nd.type_of(nd.groupby([1, 2],
                                                               ['a', 'a']))),
                        _lowlevel.type_id.GROUPBY)
        # Type
        self.assertEqual(self.type_id_of(ndt.type('type')),
                        _lowlevel.type_id.TYPE)

    def test_array_from_ptr(self):
        # cfixed_dim arrmeta is redundant so this is ok
        a = (ctypes.c_int32 * 3)()
        a[0] = 3
        a[1] = 6
        a[2] = 9
        # Readwrite version using cfixed
        b = _lowlevel.array_from_ptr('cfixed[3] * int32',
                                     ctypes.addressof(a), a, 'readwrite')
        self.assertEqual(_lowlevel.data_address_of(b), ctypes.addressof(a))
        self.assertEqual(nd.dshape_of(b), '3 * int32')
        self.assertEqual(nd.as_py(b), [3, 6, 9])
        b[1] = 10
        self.assertEqual(a[1], 10)
        # Readonly version using cfixed
        b = _lowlevel.array_from_ptr('cfixed[3] * int32',
                                     ctypes.addressof(a), a, 'readonly')
        self.assertEqual(nd.as_py(b), [3, 10, 9])
        def assign_to(b):
            b[1] = 100
        self.assertRaises(RuntimeError, assign_to, b)
        # Using a fixed dim default-constructs the arrmeta, so works too
        b = _lowlevel.array_from_ptr('3 * int32', ctypes.addressof(a),
                                     a, 'readonly')
        self.assertEqual(nd.as_py(b), [3, 10, 9])
        # Should get an error if we try strided, because the size is unknown
        self.assertRaises(RuntimeError,
                          lambda: _lowlevel.array_from_ptr('strided * int32',
                                                           ctypes.addressof(a),
                                                           a, 'readonly'))

    def test_array_from_ptr_struct(self):
        class SomeStruct(ctypes.Structure):
            _fields_ = [('x', ctypes.c_int8),
                        ('y', ctypes.c_int32),
                        ('z', ctypes.c_double)]
        a = SomeStruct()
        a.x = 1
        a.y = 2
        a.z = 3.75
        b = _lowlevel.array_from_ptr('{x: int8, y: int32, z: float64}',
                                     ctypes.addressof(a), a, 'readonly')
        self.assertEqual(nd.as_py(b, tuple=True), (1, 2, 3.75))

    def test_array_from_ptr_blockref(self):
        a = ctypes.pointer(ctypes.c_double(1.25))
        b = _lowlevel.array_from_ptr('pointer[float64]',
                                     ctypes.addressof(a), a, 'readonly')
        self.assertEqual(nd.as_py(b), 1.25)

    def test_array_from_ptr_error(self):
        # Should raise an exception if the type has arrmeta
        a = (ctypes.c_int32 * 4)()
        self.assertRaises(RuntimeError, _lowlevel.array_from_ptr,
                        ndt.type('strided * int32'), ctypes.addressof(a),
                        a, 'readwrite')

if __name__ == '__main__':
    unittest.main(verbosity=2)
