cdef extern from 'wrapper.hpp' namespace 'pydynd':
    cdef cppclass PyWrapper[T]:
        pass

    cdef void set_wrapper_type 'pydynd::set_wrapper_type'[T](object)

    cdef object wrap 'pydynd::wrap'[T](const T &)

    cdef cppclass PyWrapperIter[T]:
        pass

cdef extern from 'wrapper.hpp' namespace 'std':
    PyWrapperIter[T] begin[T](object)
    PyWrapperIter[T] end[T](object)