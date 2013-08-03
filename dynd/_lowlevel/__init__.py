from __future__ import absolute_import

from .type_id import *
from .ctypes_types import *
from .api import py_api, api
from .util import *
from .metadata_struct import build_metadata_struct

# Import all of py_api and api into the _lowlevel namespace
for name, tp in py_api._fields_:
    globals()[name] = getattr(py_api, name)
for name, tp in api._fields_:
    globals()[name] = getattr(api, name)

del py_api
del api