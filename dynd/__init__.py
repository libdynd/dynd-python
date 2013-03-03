import nd, ndt

from _pydynd import _dynd_version_string as __dynd_version_string__, \
                _dynd_python_version_string as __version_string__, \
                _dynd_git_sha1 as __dynd_git_sha1__, \
                _dynd_python_git_sha1 as __git_sha1__

__version__ = [int(x) for x in __version_string__.strip('v').split('-')[0].split('.')]
__dynd_version__ = [int(x) for x in __dynd_version_string__.strip('v').split('-')[0].split('.')]

def test(verbosity=1):
    """
    dynd.test(verbosity=1)
    
    Runs the full DyND test suite.
    """
    import os, sys, subprocess
    from tests import get_test_module
    import unittest

    print('Running unit tests for the DyND Python bindings')
    print('Python version: %s' % sys.version)
    print('Python prefix: %s' % sys.prefix)
    print('DyND-Python module: %s' % os.path.dirname(__file__))
    print('DyND-Python version: %s' % __version_string__)
    print('DyND-Python git sha1: %s' % __git_sha1__)
    print('DyND Library version: %s' % __dynd_version_string__)
    print('DyND Library git sha1: %s' % __dynd_git_sha1__)
    # Run all the tests
    all_suites = []
    for fn in os.listdir(os.path.join(os.path.dirname(__file__), 'tests')):
        if fn.startswith('test_') and fn.endswith('.py'):
            tst = get_test_module(fn[:-3])
            all_suites.append(unittest.defaultTestLoader.loadTestsFromModule(tst))
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=verbosity)
    return runner.run(unittest.TestSuite(all_suites))
