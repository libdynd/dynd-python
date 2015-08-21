from .cpp.config cimport dynd_version_string, dynd_git_sha1

cdef extern from 'exception_translation.hpp' namespace 'pydynd':
    void _translate_exception "pydynd::translate_exception"()
    void _set_broadcast_exception "pydynd::set_broadcast_exception"(object)

cdef void translate_exception():
    _translate_exception()
cdef void set_broadcast_exception(object e):
    _set_broadcast_exception(e)

cdef extern from 'do_import_array.hpp':
    pass

cdef extern from 'numpy_interop.hpp' namespace 'pydynd':
    void import_numpy()

cdef extern from 'init.hpp' namespace 'pydynd':
    void pydynd_init() except +translate_exception

cdef extern from 'git_version.hpp' namespace 'pydynd':
    extern char[] dynd_python_version_string
    extern char[] dynd_python_git_sha1

# Expose the git hashes and version numbers of this build
# NOTE: Cython generates code which is not const-correct, so
#       have to cast it away.
_dynd_version_string = str(<char *>dynd_version_string)
_dynd_git_sha1 = str(<char *>dynd_git_sha1)
_dynd_python_version_string = str(<char *>dynd_python_version_string)
_dynd_python_git_sha1 = str(<char *>dynd_python_git_sha1)

# Initialize Numpy
import_numpy()

# Exceptions to convert from C++
class BroadcastError(Exception):
    pass

# Initialize ctypes C level interop data
pydynd_init()

# Register all the exception objects with the exception translator
set_broadcast_exception(BroadcastError)
