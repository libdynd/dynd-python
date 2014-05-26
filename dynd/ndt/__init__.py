from __future__ import absolute_import, division, print_function

from dynd._pydynd import w_type as type, \
        make_byteswap, make_fixedbytes, make_convert, \
        make_view, \
        make_unaligned, make_fixedstring, make_string, \
        make_pointer, make_struct, make_cstruct, \
        make_strided_dim, make_fixed_dim, make_cfixed_dim, make_var_dim, \
        make_categorical, replace_dtype, extract_dtype, \
        factor_categorical, make_bytes, make_property, \
        make_reversed_property, cuda_support

void = type('void')
bool = type('bool')
int8 = type('int8')
int16 = type('int16')
int32 = type('int32')
int64 = type('int64')
int128 = type('int128')
intptr = type('intptr')
uint8 = type('uint8')
uint16 = type('uint16')
uint32 = type('uint32')
uint64 = type('uint64')
uint128 = type('uint128')
uintptr = type('uintptr')
float16 = type('float16')
float32 = type('float32')
float64 = type('float64')
complex_float32 = type('complex[float32]')
complex_float64 = type('complex[float64]')
# Aliases for people comfortable with the NumPy complex namings
complex64 = complex_float32
complex128 = complex_float64

string = type('string')
date = type('date')
time = type('time')
datetime = type('datetime')
datetimeutc = type('datetime[tz="UTC"]')
json = type('json')
bytes = type('bytes')

# Includes ctypes definitions
from . import dynd_ctypes as ctypes
# Some classes making dimension construction easier
from .dim_helpers import *
