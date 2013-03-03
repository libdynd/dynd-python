import nd, ndt

def test(verbosity=1):
    """
    dynd.test(verbosity=1)
    
    Runs the full DyND test suite.
    """
    import os, sys, subprocess
    from tests import get_test_module
    import unittest

    print('Running unit tests for the DyND Python bindings')
    print('Python Version: %s' % sys.version)
    print('Python Prefix: %s' % sys.prefix)
    print('DyND-Python Module: %s' % os.path.dirname(__file__))
    # Run all the tests
    all_suites = []
    for fn in os.listdir(os.path.join(os.path.dirname(__file__), 'tests')):
        if fn.startswith('test_') and fn.endswith('.py'):
            tst = get_test_module(fn[:-3])
            all_suites.append(unittest.defaultTestLoader.loadTestsFromModule(tst))
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=verbosity)
    return runner.run(unittest.TestSuite(all_suites))
    