from .type import make_fixed_bytes, make_fixed_string, make_struct, \
    make_fixed_dim, make_string, make_var_dim, make_fixed_dim_kind, \
    type_for
from .type import *

intptr = type('intptr')
uintptr = type('uintptr')
# Aliases for people comfortable with the NumPy complex namings
complex64 = complex_float32
complex128 = complex_float64

string = type('string')
#json = type('json')
bytes = type('bytes')

# Some classes making dimension construction easier
from .dim_helpers import *

from . import dynd_ctypes as ctypes

from . import json
