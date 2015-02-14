#
# Copyright (C) 2011-15 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#
# Cython notes:
#
# * Cython doesn't support 'const'. Hopefully this doesn't make it too
#   difficult to interact with const-correct code.
# * C++ 'bool' is spelled 'bint' in Cython.
# * Template functions are unsupported (classes: yes, functions: no)
# * Cython files may not contain UTF8
# * Overloading operator= is not supported
# * Note: Many things are different in Cython types, it's important to
#         read the page http://docs.cython.org/src/reference/extension_types.html,
#         as Cython does not detect incorrect usage.
# * BUG: The "except +" annotation doesn't seem to work for overloaded
#        operators, exceptions weren't being caught.

include "cpython/object.pxd"

# string.pxd does not exist in older Cython
# include "libcpp/string.pxd"
cdef extern from "<string>" namespace "std":
    cdef cppclass string:
        string(const char *)
        const char *c_str()


cdef extern from "<dynd/config.hpp>":
    # From the Cython docs:
    #   If the header file uses typedef names such as word to refer
    #   to platform-dependent flavours of numeric types, you will
    #   need a corresponding ctypedef statement, but you don't need
    #   to match the type exactly, just use something of the right
    #   general kind (int, float, etc).
    ctypedef Py_ssize_t intptr_t
    ctypedef unsigned int uintptr_t

    bint built_with_cuda "dynd::built_with_cuda" ()

cdef extern from "<dynd/config.hpp>" namespace "dynd":
    extern char[] dynd_version_string
    extern char[] dynd_git_sha1

cdef extern from "git_version.hpp" namespace "pydynd":
    extern char[] dynd_python_version_string
    extern char[] dynd_python_git_sha1

cdef extern from "<complex>" namespace "std":
    cdef cppclass complex[T]:
        T real()
        T imag()

cdef extern from "<iostream>" namespace "std":
    cdef cppclass ostream:
        pass

    extern ostream cout

cdef extern from "utility_functions.hpp" namespace "pydynd":
    object intptr_array_as_tuple(int, intptr_t *)

cdef extern from "py_lowlevel_api.hpp":
    void *dynd_get_lowlevel_api()
    void *dynd_get_py_lowlevel_api()
