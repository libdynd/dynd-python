# cython: c_string_type=str, c_string_encoding=ascii

from ..cpp.json_parser cimport discover as _discover
from ..wrapper cimport wrap

def discover(obj):
    return wrap(_discover(str(obj)))
