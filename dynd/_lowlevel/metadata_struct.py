from __future__ import absolute_import

__all__ = ['build_metadata_struct']

import ctypes

from .ctypes_types import MemoryBlockData
from .._pydynd import w_type as ndt_type

# Metadata ctypes for types that don't have child metadata
class EmptyMetadata(ctypes.Structure):
    _fields_ = []
class BytesMetadata(ctypes.Structure):
    _fields_ = [('blockref', ctypes.POINTER(MemoryBlockData))]
class StringMetadata(ctypes.Structure):
    _fields_ = [('blockref', ctypes.POINTER(MemoryBlockData))]

def build_metadata_struct(tp):
    """Builds a ctypes struct corresponding to the metadata
    for the provided dynd type. The structure is nested
    in a way which matches the nesting of the data type.

    Parameters
    ----------
    tp : dynd type
        The dynd type for which the metadata is constructed.
    """
    if not isinstance(tp, ndt_type):
        raise TypeError('tp must be a dynd type, not %r' % type(tp))
    # If there's no metadata, just return an empty struct
    if tp.metadata_size == 0:
        return EmptyMetadata
    tid = tp.type_id
    if tid == 'strided_dim':
        class StridedMetadata(ctypes.Structure):
            _fields_ = [('size', ctypes.c_ssize_t),
                        ('stride', ctypes.c_ssize_t),
                        ('element', build_metadata_struct(tp.element_type))]
        result = StridedMetadata
    elif tid == 'fixed_dim':
        class FixedMetadata(ctypes.Structure):
            _fields_ = [('element', build_metadata_struct(tp.element_type))]
        result = FixedMetadata
    elif tid == 'var_dim':
        class VarMetadata(ctypes.Structure):
            _fields_ = [('blockref', ctypes.POINTER(MemoryBlockData)),
                        ('stride', ctypes.c_ssize_t),
                        ('offset', ctypes.c_ssize_t),
                        ('element', build_metadata_struct(tp.element_type))]
        result = VarMetadata
    elif tid == 'pointer':
        class PointerMetadata(ctypes.Structure):
            _fields_ = [('blockref', ctypes.POINTER(MemoryBlockData)),
                        ('offset', ctypes.c_ssize_t),
                        ('target', build_metadata_struct(tp.target_type))]
        result = PointerMetadata
    elif tid in ['bytes', 'string', 'json']:
        result = BytesMetadata
    elif tid in ['struct', 'cstruct']:
        field_types = tp.field_types
        fields = []
        if tid == 'struct':
            # First is an array of the data offsets
            fields.append(('data_offsets',
                            ctypes.c_size_t * len(field_types)))
        # Each field metadata is stored in order
        for i, ft in enumerate(field_types):
            field_struct = build_metadata_struct(ndt_type(ft))
            fields.append(('field_%d' % i, field_struct))
        class StructMetadata(ctypes.Structure):
            _fields_ = fields
        result = StructMetadata
    elif tp.kind == 'expression':
        # For any expression type not already handled,
        # its metadata is equivalent to its operand's metadata
        result = build_metadata_struct(tp.operand_type)
    else:
        raise RuntimeError(('metadata struct for dynd type ' +
                        '%s is not handled') % tp)
    if ctypes.sizeof(result) != tp.metadata_size:
        raise RuntimeError(('internal error in metadata struct for '
                        'dynd type %s: created size %d but should be %d')
                        % (tp, ctypes.sizeof(result), tp.metadata_size))
    return result
