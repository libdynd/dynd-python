cdef extern from "dynd/complex.hpp" namespace "dynd" nogil:
    cppclass complex[T]:
        T m_real
        T m_imag
        complex()
        complex(T m_real, T m_imag)
