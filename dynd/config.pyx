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

def _get_lowlevel_api():
    return <size_t>dynd_get_lowlevel_api()

def _get_py_lowlevel_api():
    return <size_t>dynd_get_py_lowlevel_api()