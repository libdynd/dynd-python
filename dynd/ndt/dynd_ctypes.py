__all__ = ['c_dynd_bool', 'c_complex_float32', 'c_complex64',
            'c_complex_float64', 'c_complex128']

import ctypes, _ctypes
from dynd import _pydynd

class c_dynd_bool(_ctypes._SimpleCData):
    _type_ = "b"
    _dynd_type_ = _pydynd.w_type('bool')

class c_complex_float32(ctypes.Structure):
    _fields_ = [('real', ctypes.c_float),
                ('imag', ctypes.c_float)]
    _dynd_type_ = _pydynd.w_type('complex[float32]')

class c_complex_float64(ctypes.Structure):
    _fields_ = [('real', ctypes.c_double),
                ('imag', ctypes.c_double)]
    _dynd_type_ = _pydynd.w_type('complex[float64]')

c_complex64 = c_complex_float32
c_complex128 = c_complex_float64

del ctypes
del _ctypes
del _pydynd