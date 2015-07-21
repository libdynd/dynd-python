from dynd.config import *

from ..eval_context import eval_context
from .array import array, asarray, type_of, dshape_of, as_py, view, \
    ones, zeros, empty, full, is_c_contiguous, is_f_contiguous, range, \
    parse_json, squeeze, dtype_of, linspace
from .arrfunc import arrfunc

inf = float('inf')
nan = float('nan')