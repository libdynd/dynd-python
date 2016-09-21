from ..common import _load_win_dll # this also ensures that libdyndt is loaded first
import os, os.path
import sys

if os.name == 'nt':
    _load_win_dll(os.path.dirname(os.path.dirname(__file__)), 'libdynd.dll')

from dynd.config import *

from .array import array, asarray, type_of, dshape_of, as_py, view, \
    ones, zeros, empty, is_c_contiguous, is_f_contiguous, old_range, \
    parse_json, squeeze, dtype_of, old_linspace, fields, ndim_of
from .callable import callable

from . import functional

from .registry import propagate_all

propagate_all()

inf = float('inf')
nan = float('nan')
