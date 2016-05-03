import sys

from dynd.config import *

from .array import array, asarray, type_of, dshape_of, as_py, view, \
    ones, zeros, empty, is_c_contiguous, is_f_contiguous, old_range, \
    parse_json, squeeze, dtype_of, old_linspace, fields, ndim_of
from .callable import callable

inf = float('inf')
nan = float('nan')

from .registry import publish_callables
from . import functional

## This is a hack until we fix the Cython compiler issues
#class json(object):
#    @staticmethod
#    def parse(tp, obj):
#        return _parse(tp, obj)

publish_callables(sys.modules[__name__])
