from dynd.config import *

from .array import array, asarray, type_of, dshape_of, as_py, view, \
    ones, zeros, empty, full, is_c_contiguous, is_f_contiguous, range, \
    parse_json, squeeze

inf = float('inf')
nan = float('nan')