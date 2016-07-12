from libc.string cimport const_char
from libcpp.pair cimport pair
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector

from ..config cimport translate_exception
from .array cimport array
from .type cimport type

ctypedef const_char* const_charptr

cdef extern from '<sstream>' namespace 'std':
    cdef cppclass stringstream:
        string str() const

cdef extern from 'dynd/callables/base_callable.hpp' namespace 'dynd::nd' nogil:
    cdef cppclass base_callable:
        type get_type()

        const type &get_ret_type() const
        const vector[type] &get_arg_types() const
        const vector[pair[type, string]] &get_kwd_types() const

cdef extern from 'dynd/callable.hpp' namespace 'dynd::nd' nogil:
    cdef cppclass callable:
        callable()

        base_callable *get()

        bool is_null()

        array call(size_t, array *, size_t, pair[const_charptr, array] *) except +translate_exception

        base_callable &operator*() const
        array operator()(...) except +translate_exception

    callable make_callable 'dynd::nd::callable::make'[T](...) except +translate_exception

    stringstream &operator<<(callable f)

cdef extern from 'dynd/callable.hpp' namespace 'dynd' nogil:
    cdef cppclass reg_entry:
        const callable &value() const

        bool is_namespace() const

        void observe(void (*)(const char *, reg_entry *))

        map[string, reg_entry].iterator begin()
        map[string, reg_entry].iterator end()

    reg_entry &root()
    reg_entry &get(const string &, reg_entry &)
