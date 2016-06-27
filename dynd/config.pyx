from .cpp.config cimport dynd_version_string, dynd_git_sha1
from cpython.ref cimport PyObject

cdef extern from 'exception_translation.hpp':
    void _translate_exception "pydynd::translate_exception"()
    PyObject* DyND_BroadcastException

cdef void translate_exception():
    _translate_exception()

cdef extern from 'git_version.hpp' namespace 'pydynd':
    extern char[] dynd_python_version_string
    extern char[] dynd_python_git_sha1

# Expose the git hashes and version numbers of this build
# NOTE: Cython generates code which is not const-correct, so
#       have to cast it away.
_dynd_version_string = bytes(<char *>dynd_version_string).decode('ascii')
_dynd_git_sha1 = bytes(<char *>dynd_git_sha1).decode('ascii')
_dynd_python_version_string = bytes(<char *>dynd_python_version_string).decode('ascii')
_dynd_python_git_sha1 = bytes(<char *>dynd_python_git_sha1).decode('ascii')

# Exceptions to convert from C++
class BroadcastError(Exception):
    pass

# Used in exception translation header.
# It is forward-declared there and then set when this module is initialized.
DyND_BroadcastException = <PyObject*>BroadcastError

def load(name):
    _load(name)
