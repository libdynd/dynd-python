from __future__ import absolute_import

__all__ = ['array_preamble_of', 'data_address_of',
            'arrmeta_address_of', 'arrmeta_struct_of']

import ctypes

from .._pydynd import w_array as nd_array, type_of as nd_type_of
from .ctypes_types import NDArrayPreambleStruct
from .api import py_api
from .arrmeta_struct import build_arrmeta_struct

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

def arrmeta_address_of(ndo):
    """
    This low level function returns the memory address
    of the arrmeta for the array as an integer.
    """
    if not isinstance(ndo, nd_array):
        raise TypeError('Object is not a dynd array')
    # The arrmeta is always part of the array,
    # immediately after the preamble.
    return py_api.get_array_ptr(ndo) + ctypes.sizeof(NDArrayPreambleStruct)

def arrmeta_struct_of(ndo):
    """
    This low level function returns a ctypes struct
    wrapping the arrmeta of the dynd array. Note that
    this struct is not holding on to a reference to 'ndo',
    so the caller must ensure the lifetime of the arrmeta
    struct is not longer than that of 'ndo'.
    """
    if not isinstance(ndo, nd_array):
        raise TypeError('Object is not a dynd array')
    arrmeta_type = build_arrmeta_struct(nd_type_of(ndo))
    return arrmeta_type.from_address(arrmeta_address_of(ndo))

