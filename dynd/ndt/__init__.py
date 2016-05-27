import sys
import os

if os.name == 'nt':
    # Manually load dlls before loading the extension modules.
    # This is handled via rpaths on Unix based systems.
    import ctypes
    import os.path
    def load_dynd_dll(rootpath):
        try:
            ctypes.cdll.LoadLibrary(os.path.join(rootpath, 'libdyndt.dll'))
            return True
        except OSError:
            return False
    # If libdyndt.dll has been placed in the dynd-python installation, use that
    ndtdir = os.path.dirname(os.path.dirname(__file__))
    loaded = load_dynd_dll(ndtdir)
    # Next, try the default DLL search path
    loaded = loaded or load_dynd_dll('')
    if not loaded:
        # Try to load it from the Program Files directories where libdynd
        # installs by default. This matches the search path for libdynd used
        # in the CMake build for dynd-python.
        is_64_bit = sys.maxsize > 2**32
        processor_arch = os.environ.get('PROCESSOR_ARCHITECTURE')
        err_str = ('Fallback search for libdyndt.dll failed because the "{}" '
                   'environment variable was not set. Please make sure that '
                   'either libdyndt is on the DLL search path or that it is '
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
                raise ctypes.WinError(126, 'Could not load libdyndt.dll')

from .type import make_fixed_bytes, make_fixed_string, make_struct, \
    make_fixed_dim, make_string, make_var_dim, make_fixed_dim_kind, \
    type_for
from .type import *

# Some classes making dimension construction easier
from .dim_helpers import *

from . import dynd_ctypes as ctypes

from . import json

del os
del sys
