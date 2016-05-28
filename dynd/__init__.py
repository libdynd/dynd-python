# DyND is a namespace package with the components dynd.ndt and dynd.nd.
try:
    import pkg_resources
    pkg_resources.declare_namespace(__name__)
except ImportError:
    import pkgutil
    __path__ = pkgutil.extend_path(__path__, __name__)

# The 'common' import is only supposed to work from the dynd.ndt sub-package.
# If dynd.nd/dynd/__init__.py and dynd.ndt/dynd/__init__.py are the same file,
# this ImportError must be ignored. The alternative is to ship two different
# __init__.py files and remove the import from dynd.nd/dynd/__init__.py.
try:
    from .common import *
except ImportError:
    pass

__all__ = [
    '__libdynd_version__', '__version__', '__libdynd_git_sha1__', '__git_sha1__',
    'annotate', 'test', 'load'
]

