import nd, ndt

from _pydynd import _dynd_version_string as __dynd_version_string__, \
                _dynd_python_version_string as __version_string__, \
                _dynd_git_sha1 as __dynd_git_sha1__, \
                _dynd_python_git_sha1 as __git_sha1__

__version__ = [int(x) for x in __version_string__.strip('v').split('-')[0].split('.')]
__dynd_version__ = [int(x) for x in __dynd_version_string__.strip('v').split('-')[0].split('.')]

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
    from tests import get_tst_module
    import unittest

    print('Running unit tests for the DyND Python bindings')
    print('Python version: %s' % sys.version)
    print('Python prefix: %s' % sys.prefix)
    print('DyND-Python module: %s' % os.path.dirname(__file__))
    print('DyND-Python version: %s' % __version_string__)
    print('DyND-Python git sha1: %s' % __git_sha1__)
    print('DyND Library version: %s' % __dynd_version_string__)
    print('DyND Library git sha1: %s' % __dynd_git_sha1__)
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
        # Use nose to run the tests and produce an XML file
        import nose
        return nose.main(argv=['nosetests',
                        '--with-xunit',
                        '--xunit-file=%s' % xunitfile,
                        os.path.join(os.path.dirname(__file__), 'tests')],
                     exit=exit)
