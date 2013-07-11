from __future__ import absolute_import

from dynd._pydynd import w_type as type, \
        make_byteswap_dtype, make_fixedbytes_dtype, make_convert_dtype, \
        make_view_dtype, \
        make_unaligned_dtype, make_fixedstring_type, make_string_type, \
        make_pointer_dtype, make_struct_dtype, make_cstruct_dtype, \
        make_strided_dim_dtype, make_fixed_dim_dtype, make_var_dim_dtype, \
        make_categorical_dtype, replace_udtype, extract_udtype, \
        factor_categorical_dtype, make_bytes_dtype

void = type('void')
bool = type('bool')
int8 = type('int8')
int16 = type('int16')
int32 = type('int32')
int64 = type('int64')
uint8 = type('uint8')
uint16 = type('uint16')
uint32 = type('uint32')
uint64 = type('uint64')
float32 = type('float32')
float64 = type('float64')
complex_float32 = type('cfloat32')
complex_float64 = type('cfloat64')
cfloat32 = complex_float32
cfloat64 = complex_float64
# Aliases for people comfortable with the Numpy complex namings
complex64 = cfloat32
complex128 = cfloat64

string = type('string')
date = type('date')
json = type('json')
bytes = type('bytes')

# Includes ctypes definitions
from . import dynd_ctypes as ctypes
