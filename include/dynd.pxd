#
# Copyright (C) 2011-13, DyND Developers
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
        char *c_str()

cdef extern from "<dynd/config.hpp>":
    # From the Cython docs:
    #   If the header file uses typedef names such as word to refer
    #   to platform-dependent flavours of numeric types, you will
    #   need a corresponding ctypedef statement, but you don't need
    #   to match the type exactly, just use something of the right
    #   general kind (int, float, etc).
    ctypedef Py_ssize_t intptr_t
    ctypedef unsigned int uintptr_t

cdef extern from "<complex>" namespace "std":
    cdef cppclass complex[T]:
        T real()
        T imag()

cdef extern from "<iostream>" namespace "std":
    cdef cppclass ostream:
        pass

    extern ostream cout

cdef extern from "<dynd/json_parser.hpp>" namespace "dynd":
    ndobject dynd_parse_json "dynd::parse_json" (dtype&, ndobject&) except +translate_exception

cdef extern from "utility_functions.hpp" namespace "pydynd":
    object intptr_array_as_tuple(int, intptr_t *)

cdef extern from "placement_wrappers.hpp" namespace "pydynd":
    cdef struct dtype_placement_wrapper:
        pass
    void placement_new(dtype_placement_wrapper&) except +translate_exception
    void placement_delete(dtype_placement_wrapper&)
    # dtype placement cast
    dtype& GET(dtype_placement_wrapper&)
    # dtype placement assignment
    void SET(dtype_placement_wrapper&, dtype&)

    cdef struct ndobject_placement_wrapper:
        pass
    void placement_new(ndobject_placement_wrapper&) except +translate_exception
    void placement_delete(ndobject_placement_wrapper&)
    # ndobject placement cast
    ndobject& GET(ndobject_placement_wrapper&)
    # ndobject placement assignment
    void SET(ndobject_placement_wrapper&, ndobject&)

    cdef struct codegen_cache_placement_wrapper:
        pass
    void placement_new(codegen_cache_placement_wrapper&) except +translate_exception
    void placement_delete(codegen_cache_placement_wrapper&)
    # placement cast
    codegen_cache& GET(codegen_cache_placement_wrapper&)

    cdef struct vm_elwise_program_placement_wrapper:
        pass
    void placement_new(vm_elwise_program_placement_wrapper&) except +translate_exception
    void placement_delete(vm_elwise_program_placement_wrapper&)
    # placement cast
    elwise_program& GET(vm_elwise_program_placement_wrapper&)
    void SET(vm_elwise_program_placement_wrapper&, elwise_program&)
