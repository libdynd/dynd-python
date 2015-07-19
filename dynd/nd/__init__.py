from __future__ import absolute_import

# Expose types and functions directly from the Cython/C++ module
from .._pydynd import w_array as array, w_arrfunc as arrfunc, \
        w_eval_context as eval_context, \
        as_py, as_numpy, zeros, ones, full, empty, empty_like, range, \
        linspace, memmap, fields, groupby, \
        parse_json, format_json, debug_repr, \
        BroadcastError, type_of, dtype_of, dshape_of, ndim_of, \
        view, adapt, asarray, is_c_contiguous, is_f_contiguous, \
        rolling_apply, modify_default_eval_context

from .gfunc import *

from .computed_fields import add_computed_fields, make_computed_fields
from .array_functions import squeeze
from .functional import inline

inf = float('inf')
nan = float('nan')
