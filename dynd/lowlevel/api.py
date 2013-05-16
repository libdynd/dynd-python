from __future__ import absolute_import

__all__ = ['api', 'py_api']

import ctypes
from .ctypes_types import LowLevelAPI, PyLowLevelAPI
from dynd._pydynd import _get_lowlevel_api, _get_py_lowlevel_api

api = LowLevelAPI.from_address(_get_lowlevel_api())
py_api = PyLowLevelAPI.from_address(_get_py_lowlevel_api())
