cdef extern from 'wrapper.hpp':
    cdef cppclass wrapper 'DyND_PyWrapper'[T]:
        pass

    cdef void set_wrapper_type 'DyND_PyWrapper_Type'[T](object)

    cdef object wrap 'DyND_PyWrapper_New'[T](const T &)

    cdef cppclass wrapper_iter 'DyND_PyWrapperIter'[T]:
        pass

cdef extern from 'wrapper.hpp' namespace 'std':
    wrapper_iter[T] begin[T](object)
    wrapper_iter[T] end[T](object)