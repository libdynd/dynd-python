from ..config cimport translate_exception

cdef extern from 'assign.hpp':
    void assign_init() except +translate_exception
