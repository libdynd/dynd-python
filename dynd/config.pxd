from libcpp.string cimport string

from translate_except cimport translate_exception, set_broadcast_exception

cdef extern from "do_import_array.hpp":
    pass
cdef extern from "numpy_interop.hpp" namespace "pydynd":
    void import_numpy()

cdef extern from "init.hpp" namespace "pydynd":
    void pydynd_init() except +translate_exception

from dynd.ndt.type cimport _type
from dynd.nd.array cimport _array

cdef extern from "numpy_interop.hpp" namespace "pydynd":
    object array_as_numpy_struct_capsule(_array&) except +translate_exception

cdef extern from "<dynd/json_formatter.hpp>" namespace "dynd":
    _array dynd_format_json "dynd::format_json" (_array&, bint) except +translate_exception

cdef extern from "<dynd/types/datashape_formatter.hpp>" namespace "dynd":
    string dynd_format_datashape "dynd::format_datashape" (_array&) except +translate_exception
    string dynd_format_datashape "dynd::format_datashape" (_type&) except +translate_exception

cdef class w_type:
    cdef _type v

cdef class w_array:
    cdef _array v