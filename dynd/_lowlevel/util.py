from __future__ import absolute_import

__all__ = ['array_preamble_of', 'data_address_of',
            'metadata_address_of', 'metadata_struct_of']

import ctypes

from .._pydynd import w_array as nd_array, type_of as nd_type_of
from .ctypes_types import NDArrayPreambleStruct
from .api import py_api
from .metadata_struct import build_metadata_struct

def array_preamble_of(ndo):
    """
    This low level function returns a ctypes structure
    exposing the preamble of the dynd array. Note that
    this struct is not holding on to a reference to 'ndo',
    so the caller must ensure the lifetime of the
    structure is not longer than that of 'ndo'.
    """
    if not isinstance(ndo, nd_array):
        raise TypeError('Object is not a dynd array')
    return NDArrayPreambleStruct.from_address(py_api.get_array_ptr(ndo))

def data_address_of(ndo):
    """
    This low level function returns the memory address
    of the data for the array as an integer.
    """
    if not isinstance(ndo, nd_array):
        raise TypeError('Object is not a dynd array')
    s = NDArrayPreambleStruct.from_address(py_api.get_array_ptr(ndo))
    return s.data_pointer

def metadata_address_of(ndo):
    """
    This low level function returns the memory address
    of the metadata for the array as an integer.
    """
    if not isinstance(ndo, nd_array):
        raise TypeError('Object is not a dynd array')
    # The metadata is always part of the array,
    # immediately after the preamble.
    return py_api.get_array_ptr(ndo) + ctypes.sizeof(NDArrayPreambleStruct)

def metadata_struct_of(ndo):
    """
    This low level function returns a ctypes struct
    wrapping the metadata of the dynd array. Note that
    this struct is not holding on to a reference to 'ndo',
    so the caller must ensure the lifetime of the metadata
    struct is not longer than that of 'ndo'.
    """
    if not isinstance(ndo, nd_array):
        raise TypeError('Object is not a dynd array')
    metadata_type = build_metadata_struct(nd_type_of(ndo))
    return metadata_type.from_address(metadata_address_of(ndo))

