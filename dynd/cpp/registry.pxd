from libc.string cimport const_char
from libcpp.pair cimport pair
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector

from .callable cimport callable
from ..config cimport translate_exception

cdef extern from 'dynd/registry.hpp' namespace 'dynd' nogil:
    cdef cppclass registry_entry:
        const callable &value() const

        bool is_namespace() const

        void observe(void (*)(registry_entry *, const char *, registry_entry *))

        registry_entry &get(const string &)

        const string &path() const

        map[string, registry_entry].iterator begin()
        map[string, registry_entry].iterator end()

    registry_entry &registered()
    registry_entry &registered(const string &)
