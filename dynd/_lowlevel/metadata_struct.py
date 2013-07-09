from __future__ import absolute_import

__all__ = ['build_metadata_struct']

import ctypes

from .ctypes_types import MemoryBlockData
from ..ndt import type as ndt_type

# Metadata ctypes for types that don't have child metadata
class EmptyMetadata(ctypes.Structure):
    _fields_ = []
class BytesMetadata(ctypes.Structure):
    _fields_ = [('blockref', ctypes.POINTER(MemoryBlockData))]
class StringMetadata(ctypes.Structure):
    _fields_ = [('blockref', ctypes.POINTER(MemoryBlockData))]

def build_metadata_struct(dt):
    """Builds a ctypes struct corresponding to the metadata
    for the provided dynd type. The structure is nested
    in a way which matches the nesting of the data type.

    Parameters
    ----------
    dt : dynd type
        The dynd type for which the metadata is constructed.
    """
    if not isinstance(dt, ndt_type):
        raise TypeError('dt must be a dynd type, not %r' % type(dt))
    # If there's no metadata, just return an empty struct
    if dt.metadata_size == 0:
        return EmptyMetadata
    tid = dt.type_id
    if tid == 'strided_dim':
        class StridedMetadata(ctypes.Structure):
            _fields_ = [('size', ctypes.c_ssize_t),
                        ('stride', ctypes.c_ssize_t),
                        ('element', build_metadata_struct(dt.element_type))]
        result = StridedMetadata
    elif tid == 'fixed_dim':
        class FixedMetadata(ctypes.Structure):
            _fields_ = [('element', build_metadata_struct(dt.element_type))]
        result = FixedMetadata
    elif tid == 'var_dim':
        class VarMetadata(ctypes.Structure):
            _fields_ = [('blockref', ctypes.POINTER(MemoryBlockData)),
                        ('stride', ctypes.c_ssize_t),
                        ('offset', ctypes.c_ssize_t),
                        ('element', build_metadata_struct(dt.element_type))]
        result = VarMetadata
    elif tid == 'pointer':
        class PointerMetadata(ctypes.Structure):
            _fields_ = [('blockref', ctypes.POINTER(MemoryBlockData)),
                        ('offset', ctypes.c_ssize_t),
                        ('target', build_metadata_struct(dt.target_type))]
        result = PointerMetadata
    elif tid in ['bytes', 'string', 'json']:
        result = BytesMetadata
    elif tid in ['struct', 'cstruct']:
        field_types = dt.field_types
        fields = []
        if tid == 'struct':
            # First is an array of the data offsets
            fields.append(('data_offsets',
                            ctypes.c_size_t * len(field_types)))
        # Each field metadata is stored in order
        for i, ft in enumerate(field_types):
            fields.append(('field_%d' % i, build_metadata_struct(ft)))
        class StructMetadata(ctypes.Structure):
            _fields_ = fields
        result = StructMetadata
    elif kind == 'expression':
        # For any expression type not already handled,
        # its metadata is equivalent to its operand's metadata
        result = build_metadata_struct(dt.operand_type)
    else:
        raise RuntimeError(('metadata struct for dynd type ' +
                        '%s is not handled') % dt)
    if ctypes.sizeof(result) != dt.metadata_size:
        raise RuntimeError(('internal error in metadata struct for '
                        'dynd type %s: created size %d but should be %d')
                        % (dt, ctypes.sizeof(result), dt.metadata_size))
    return result
