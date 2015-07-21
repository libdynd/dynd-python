from .type import type, Unsupplied, make_fixed_bytes, make_fixed_string, make_struct, \
    make_fixed_dim, make_string, make_var_dim, make_fixed_dim_kind, make_byteswap, \
    make_unaligned, make_convert

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

# Some classes making dimension construction easier
from .dim_helpers import *

from . import dynd_ctypes as ctypes