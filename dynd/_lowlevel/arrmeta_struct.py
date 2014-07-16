from __future__ import absolute_import

__all__ = ['build_arrmeta_struct']

import ctypes

from .ctypes_types import MemoryBlockData
from .._pydynd import w_type as ndt_type

# Metadata ctypes for types that don't have child arrmeta
class EmptyMetadata(ctypes.Structure):
    _fields_ = []
class BytesMetadata(ctypes.Structure):
    _fields_ = [('blockref', ctypes.POINTER(MemoryBlockData))]
class StringMetadata(ctypes.Structure):
    _fields_ = [('blockref', ctypes.POINTER(MemoryBlockData))]

def build_arrmeta_struct(tp):
    """Builds a ctypes struct corresponding to the arrmeta
    for the provided dynd type. The structure is nested
    in a way which matches the nesting of the data type.

    Parameters
    ----------
    tp : dynd type
        The dynd type for which the arrmeta is constructed.
    """
    if not isinstance(tp, ndt_type):
        raise TypeError('tp must be a dynd type, not %r' % type(tp))
    # If there's no arrmeta, just return an empty struct
    if tp.arrmeta_size == 0:
        return EmptyMetadata
    tid = tp.type_id
    if tid == 'strided_dim':
        class StridedDimMetadata(ctypes.Structure):
            _fields_ = [('dim_size', ctypes.c_ssize_t),
                        ('stride', ctypes.c_ssize_t),
                        ('element', build_arrmeta_struct(tp.element_type))]
        result = StridedDimMetadata
    elif tid == 'fixed_dim':
        class FixedDimMetadata(ctypes.Structure):
            _fields_ = [('dim_size', ctypes.c_ssize_t),
                        ('stride', ctypes.c_ssize_t),
                        ('element', build_arrmeta_struct(tp.element_type))]
        result = FixedDimMetadata
    elif tid == 'cfixed_dim':
        class CFixedDimMetadata(ctypes.Structure):
            _fields_ = [('dim_size', ctypes.c_ssize_t),
                        ('stride', ctypes.c_ssize_t),
                        ('element', build_arrmeta_struct(tp.element_type))]
        result = CFixedDimMetadata
    elif tid == 'var_dim':
        class VarMetadata(ctypes.Structure):
            _fields_ = [('blockref', ctypes.POINTER(MemoryBlockData)),
                        ('stride', ctypes.c_ssize_t),
                        ('offset', ctypes.c_ssize_t),
                        ('element', build_arrmeta_struct(tp.element_type))]
        result = VarMetadata
    elif tid == 'pointer':
        class PointerMetadata(ctypes.Structure):
            _fields_ = [('blockref', ctypes.POINTER(MemoryBlockData)),
                        ('offset', ctypes.c_ssize_t),
                        ('target', build_arrmeta_struct(tp.target_type))]
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
        # Each field arrmeta is stored in order
        for i, ft in enumerate(field_types):
            field_struct = build_arrmeta_struct(ndt_type(ft))
            fields.append(('field_%d' % i, field_struct))
        class StructMetadata(ctypes.Structure):
            _fields_ = fields
        result = StructMetadata
    elif tp.kind == 'expression':
        # For any expression type not already handled,
        # its arrmeta is equivalent to its operand's arrmeta
        result = build_arrmeta_struct(tp.operand_type)
    else:
        raise RuntimeError(('arrmeta struct for dynd type ' +
                        '%s is not handled') % tp)
    if ctypes.sizeof(result) != tp.arrmeta_size:
        raise RuntimeError(('internal error in arrmeta struct for '
                        'dynd type %s: created size %d but should be %d')
                        % (tp, ctypes.sizeof(result), tp.arrmeta_size))
    return result
