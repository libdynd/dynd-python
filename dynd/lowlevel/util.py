from __future__ import absolute_import

__all__ = ['data_address_of', 'metadata_address_of']

from dynd import nd, ndt
from .ctypes_types import NDObjectPreamble
from .api import py_api

def data_address_of(ndo):
    """
    This low level function returns the memory address
    of the data for the ndobject as an integer.
    """
    if not isinstance(ndo, nd.ndobject):
        raise TypeError('Object is not a dynd ndobject')
    s = NDObjectPreamble.from_address(py_api.get_ndobject_ptr(ndo))
    return s.data_pointer

def metadata_address_of(ndo):
    """
    This low level function returns the memory address
    of the metadata for the ndobject as an integer.
    """
    if not isinstance(ndo, nd.ndobject):
        raise TypeError('Object is not a dynd ndobject')
    # The metadata is always part of the ndobject,
    # immediately after the preamble.
    return py_api.get_ndobject_ptr(ndo) + ctypes.sizeof(NDObjectPreamble)
