from __future__ import absolute_import

# dynd._lowlevel is not imported by default
from . import nd, ndt

from ._pydynd import _dynd_version_string as __libdynd_version__, \
                _dynd_python_version_string as __version__, \
                _dynd_git_sha1 as __libdynd_git_sha1__, \
                _dynd_python_git_sha1 as __git_sha1__

def fix_version(v):
    if '.' in v:
        vlst = v.lstrip('v').split('.')
        vlst = vlst[:-1] + vlst[-1].split('-')

        if len(vlst) <= 3:
            vtup = tuple(int(x) for x in vlst)
        else:
            # The first 3 numbers are always integer
            vtup = tuple(int(x) for x in vlst[:3])
            # The 4th one may not be, so trap it
            try:
                vtup = vtup + (int(vlst[3]),)
                # Zero pad the post version #, so it sorts lexicographically
                vlst[3] = 'post%03d' % int(vlst[3])
            except ValueError:
                pass
        return '.'.join(vlst), vtup
    else:
        # When a "git checkout --depth=1 ..." has been done,
        # it will look like "96da079" or "96da079-dirty"
        return v, (0, 0, 0)

__version__, __version_info__ = fix_version(__version__)
__libdynd_version__, __libdynd_version_info__ = fix_version(__libdynd_version__)

del fix_version

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
