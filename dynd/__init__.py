from __future__ import absolute_import

# dynd._lowlevel is not imported by default
from . import nd, ndt

from ._pydynd import _dynd_version_string as __libdynd_version__, \
                _dynd_python_version_string as __version__, \
                _dynd_git_sha1 as __libdynd_git_sha1__, \
                _dynd_python_git_sha1 as __git_sha1__

__version__ = __version__.lstrip('v')
__libdynd_version__ = __libdynd_version__.lstrip('v')
__version_info__ = tuple(int(x)
                for x in __version__.split('-')[0].split('.'))
__libdynd_version_info__ = tuple(int(x)
                for x in __libdynd_version__.split('-')[0].split('.'))

def test(verbosity=1, xunitfile=None, exit=False):
    """
    Runs the full DyND test suite, outputing
    the results of the tests to  sys.stdout.

    Parameters
    ----------
    verbosity : int, optional
        Value 0 prints very little, 1 prints a little bit,
        and 2 prints the test names while testing.
    xunitfile : string, optional
        If provided, writes the test results to an xunit
        style xml file. This is useful for running the tests
        in a CI server such as Jenkins.
    exit : bool, optional
        If True, the function will call sys.exit with an
        error code after the tests are finished.
    """
    import os, sys, subprocess
    import numpy
    from .tests import get_tst_module
    import unittest

    print('Running unit tests for the DyND Python bindings')
    print('Python version: %s' % sys.version)
    print('Python prefix: %s' % sys.prefix)
    print('DyND-Python module: %s' % os.path.dirname(__file__))
    print('DyND-Python version: %s' % __version__)
    print('DyND-Python git sha1: %s' % __git_sha1__)
    print('LibDyND version: %s' % __libdynd_version__)
    print('LibDyND git sha1: %s' % __libdynd_git_sha1__)
    print('NumPy version: %s' % numpy.__version__)
    sys.stdout.flush()
    if xunitfile is None:
        # Run all the tests
        all_suites = []
        for fn in os.listdir(os.path.join(os.path.dirname(__file__), 'tests')):
            if fn.startswith('test_') and fn.endswith('.py'):
                tst = get_tst_module(fn[:-3])
                all_suites.append(unittest.defaultTestLoader.loadTestsFromModule(tst))
        runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=verbosity)
        result = runner.run(unittest.TestSuite(all_suites))
        if exit:
            sys.exit(not result.wasSuccessful())
        else:
            return result
    else:
        import nose
        import os
        argv = ['nosetests', '--verbosity=%d' % verbosity]
        # Output an xunit file if requested
        if xunitfile:
            argv.extend(['--with-xunit', '--xunit-file=%s' % xunitfile])
        # Add all 'tests' subdirectories to the options
        rootdir = os.path.dirname(__file__)
        for root, dirs, files in os.walk(rootdir):
            if 'tests' in dirs:
                testsdir = os.path.join(root, 'tests')
                argv.append(testsdir)
                print('Test dir: %s' % testsdir[len(rootdir)+1:])
        # Ask nose to do its thing
        return nose.main(argv=argv, exit=exit)
