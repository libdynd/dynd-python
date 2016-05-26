import dynd.ndt
import sys
import os

if os.name == 'nt':
    # Manually load dlls before loading the extension modules.
    # This is handled via rpaths on Unix based systems.
    import ctypes
    import os.path
    def load_dynd_dll(rootpath):
        try:
            ctypes.cdll.LoadLibrary(os.path.join(rootpath, 'libdynd.dll'))
            return True
        except OSError:
            return False
    # If libdynd.dll has been placed in the dynd-python installation, use that
    nddir = os.path.dirname(os.path.dirname(__file__))
    loaded = load_dynd_dll(nddir)
    # Next, try the default DLL search path
    loaded = loaded or load_dynd_dll('')
    if not loaded:
        # Try to load it from the Program Files directories where libdynd
        # installs by default. This matches the search path for libdynd used
        # in the CMake build for dynd-python.
        is_64_bit = sys.maxsize > 2**32
        processor_arch = os.environ.get('PROCESSOR_ARCHITECTURE')
        err_str = ('Fallback search for libdynd.dll failed because the "{}" '
                   'environment variable was not set. Please make sure that '
                   'either libdynd is on the DLL search path or that it is '
                   'in the default install directory and the runtime '
                   'environment has the necessary system-specified '
                   'environment variables properly set. On 64 bit Windows '
                   'with 64 bit Python the needed variables are '
                   '"PROCESSOR_ARCHITECTURE" and "ProgramFiles". On 64 bit '
                   'Windows with 32 bit Python the needed variables are '
                   '"PROCESSOR_ARCHITECTURE" and "ProgramFiles(x86)". On 32 '
                   'bit Windows the needed variables are '
                   '"PROCESSOR_ARCHITECTURE" and "ProgramFiles".')
        if processor_arch is None:
            raise RuntimeError(err_str.format('PROCESSOR_ARCHITECTURE'))
        is_32_on_64_bit = (is_64_bit and not processor_arch.endswith('64'))
        if not is_32_on_64_bit:
            prog_files = os.environ.get('ProgramFiles')
            if prog_files is None:
                raise RuntimeError(err_str.format('ProgramFiles'))
        else:
            prog_files = os.environ.get('ProgramFiles(x86)')
            if prog_files is None:
                raise RuntimeError(err_str.format('ProgramFiles(x86)'))
        dynd_lib_dir = os.path.join(prog_files, 'libdynd', 'lib')
        if os.path.isdir(dynd_lib_dir):
            loaded = load_dynd_dll(dynd_lib_dir)
            if not loaded:
                raise ctypes.WinError(126, 'Could not load libdynd.dll')


from dynd.config import *

from .array import array, asarray, type_of, dshape_of, as_py, view, \
    ones, zeros, empty, is_c_contiguous, is_f_contiguous, old_range, \
    parse_json, squeeze, dtype_of, old_linspace, fields, ndim_of
from .callable import callable

inf = float('inf')
nan = float('nan')

from .registry import publish_callables
from . import functional

## This is a hack until we fix the Cython compiler issues
#class json(object):
#    @staticmethod
#    def parse(tp, obj):
#        return _parse(tp, obj)

publish_callables(sys.modules[__name__])

del os
del sys
del dynd.ndt
