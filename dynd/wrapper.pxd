cdef extern from 'wrapper.hpp':
    cdef cppclass wrapper 'DyND_PyWrapper'[T]:
        pass

    cdef object wrap 'DyND_PyWrapper_New'[T](const T &)