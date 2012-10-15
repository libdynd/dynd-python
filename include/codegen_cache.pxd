#
# Copyright (C) 2011-12, Dynamic NDArray Developers
# BSD 2-Clause License, see LICENSE.txt
#

cdef extern from "dnd/codegen/codegen_cache.hpp" namespace "dynd":
    cdef cppclass codegen_cache:
        pass

cdef extern from "codegen_cache_functions.hpp" namespace "pydynd":
    string codegen_cache_debug_dump(codegen_cache&) except +translate_exception
