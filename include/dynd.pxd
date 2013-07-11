#
# Copyright (C) 2011-13 Mark Wiebe, DyND Developers
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

cdef extern from "<dynd/json_parser.hpp>" namespace "dynd":
    ndarray dynd_parse_json_type "dynd::parse_json" (ndt_type&, ndarray&) except +translate_exception
    void dynd_parse_json_array "dynd::parse_json" (ndarray&, ndarray&) except +translate_exception

cdef extern from "<dynd/json_formatter.hpp>" namespace "dynd":
    ndarray dynd_format_json "dynd::format_json" (ndarray&) except +translate_exception

cdef extern from "<dynd/types/datashape_formatter.hpp>" namespace "dynd":
    string dynd_format_datashape "dynd::format_datashape" (ndarray&) except +translate_exception
    string dynd_format_datashape "dynd::format_datashape" (ndt_type&) except +translate_exception

cdef extern from "utility_functions.hpp" namespace "pydynd":
    object intptr_array_as_tuple(int, intptr_t *)

cdef extern from "placement_wrappers.hpp" namespace "pydynd":
    cdef struct ndt_type_placement_wrapper:
        pass
    void placement_new(ndt_type_placement_wrapper&) except +translate_exception
    void placement_delete(ndt_type_placement_wrapper&)
    # type placement cast
    ndt_type& GET(ndt_type_placement_wrapper&)
    # type placement assignment
    void SET(ndt_type_placement_wrapper&, ndt_type&)

    cdef struct array_placement_wrapper:
        pass
    void placement_new(array_placement_wrapper&) except +translate_exception
    void placement_delete(array_placement_wrapper&)
    # nd::array placement cast
    ndarray& GET(array_placement_wrapper&)
    # nd::array placement assignment
    void SET(array_placement_wrapper&, ndarray&)

#    cdef struct codegen_cache_placement_wrapper:
#        pass
#    void placement_new(codegen_cache_placement_wrapper&) except +translate_exception
#    void placement_delete(codegen_cache_placement_wrapper&)
#    # placement cast
#    codegen_cache& GET(codegen_cache_placement_wrapper&)

    cdef struct vm_elwise_program_placement_wrapper:
        pass
    void placement_new(vm_elwise_program_placement_wrapper&) except +translate_exception
    void placement_delete(vm_elwise_program_placement_wrapper&)
    # placement cast
    elwise_program& GET(vm_elwise_program_placement_wrapper&)
    void SET(vm_elwise_program_placement_wrapper&, elwise_program&)

cdef extern from "py_lowlevel_api.hpp":
    void *dynd_get_lowlevel_api()
    void *dynd_get_py_lowlevel_api()
