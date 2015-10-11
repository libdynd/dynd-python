from .type import Unsupplied, make_fixed_bytes, make_fixed_string, make_struct, \
    make_fixed_dim, make_string, make_var_dim, make_fixed_dim_kind, make_byteswap, \
    make_unaligned, make_convert, make_categorical, make_view
from .type import *

intptr = type('intptr')
uintptr = type('uintptr')
# Aliases for people comfortable with the NumPy complex namings
complex64 = complex_float32
complex128 = complex_float64

string = type('string')
date = type('date')
time = type('time')
datetime = type('datetime')
datetimeutc = type('datetime[tz="UTC"]')
#json = type('json')
bytes = type('bytes')

# Some classes making dimension construction easier
from .dim_helpers import *

from . import dynd_ctypes as ctypes

from . import json
