# DyND is a namespace package with the components dynd.ndt and dynd.nd.
try:
    import pkg_resources
    pkg_resources.declare_namespace(__name__)
except ImportError:
    import pkgutil
    __path__ = pkgutil.extend_path(__path__, __name__)

try: # Only the import from dynd.ndt is expected to succeed.
    from .common import *
except ImportError:
    pass

__all__ = [
    '__libdynd_version__', '__version__', '__libdynd_git_sha1__', ' __git_sha1__',
    'annotate', 'test', 'load'
]


try: del common
except: pass

try: del pkgutil
except: pass

try: del pkg_resources
except: pass


