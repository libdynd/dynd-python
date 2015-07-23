cdef extern from 'wrapper.hpp':
    cdef cppclass wrapper 'DyND_PyWrapper'[T]:
        pass

    cdef void set_wrapper_type 'DyND_PyWrapper_Type'[T](object)

    cdef object wrap 'DyND_PyWrapper_New'[T](const T &)