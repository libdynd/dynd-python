from __future__ import absolute_import

from dynd._pydynd import w_type as type, \
        make_byteswap, make_fixedbytes, make_convert, \
        make_view, \
        make_unaligned, make_fixedstring, make_string, \
        make_pointer, make_struct, make_cstruct, \
        make_strided_dim, make_fixed_dim, make_var_dim, \
        make_categorical, replace_dtype, extract_dtype, \
        factor_categorical, make_bytes

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
