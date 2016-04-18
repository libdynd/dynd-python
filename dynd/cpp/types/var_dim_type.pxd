from ..type cimport type

from ...config cimport translate_exception

cdef extern from 'dynd/types/var_dim_type.hpp' namespace 'dynd::ndt':
    cppclass var_dim_type:
        pass
