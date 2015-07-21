cdef extern from 'exception_translation.hpp' namespace 'pydynd':
    void translate_exception()
    void set_broadcast_exception(object)

cdef extern from 'do_import_array.hpp':
    pass

cdef extern from 'numpy_interop.hpp' namespace 'pydynd':
    void import_numpy()

cdef extern from 'init.hpp' namespace 'pydynd':
    void pydynd_init() except +translate_exception

cdef extern from 'dynd/config.hpp' namespace 'dynd':
    extern char[] dynd_version_string
    extern char[] dynd_git_sha1

cdef extern from 'git_version.hpp' namespace 'pydynd':
    extern char[] dynd_python_version_string
    extern char[] dynd_python_git_sha1

cdef extern from "py_lowlevel_api.hpp":
    void *dynd_get_lowlevel_api()
    void *dynd_get_py_lowlevel_api()