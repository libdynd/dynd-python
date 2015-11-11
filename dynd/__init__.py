import os
if os.name == 'nt':
    # Manually load dlls before loading the extension modules.
    # TODO: template a file using cmake that stores the location of the
    # libdynd library used during compilation.
    import os.path
    import sys
    from ctypes import cdll
    is_64_bit = sys.maxsize > 2**32
    pydynd_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isfile(os.path.join(pydynd_dir, 'libdynd.dll')):
        if is_64_bit:
            libdynd_path = os.path.join(os.environ['ProgramFiles'], 'libdynd',
                                        'lib', 'libdynd.dll')
            if os.path.isfile(libdynd_path):
                cdll.LoadLibrary(libdynd_path)
        else:
            libdynd_path = os.path.join(os.environ['ProgramFiles(x86)'],
                                        'libdynd', 'lib', 'libdynd.dll')
            if os.path.isfile(libdynd_path):
                cdll.LoadLibrary(libdynd_path)
    else:
        cdll.LoadLibrary(os.path.join(pydynd_dir, 'libdynd.dll'))
    cdll.LoadLibrary(os.path.join(pydynd_dir, 'pydynd.dll'))

from .config import _dynd_version_string as __libdynd_version__, \
                _dynd_python_version_string as __version__, \
                _dynd_git_sha1 as __libdynd_git_sha1__, \
                _dynd_python_git_sha1 as __git_sha1__

def annotate(*args, **kwds):
    def wrap(func):
        func.__annotations__ = {}

        try:
            func.__annotations__['return'] = args[0]
        except IndexError:
            pass

        if len(args[1:]) > func.__code__.co_argcount:
            raise TypeError('{0} takes {1} positional arguments but {2} positional annotations were given'.format(func,
                func.__code__.co_argcount, len(args) - 1))

        for key, value in zip(func.__code__.co_varnames, args[1:]):
            func.__annotations__[key] = value

        for key, value in kwds.items():
            if key not in func.__code__.co_varnames:
                raise TypeError("{0} got an unexpected keyword annotation '{1}'".format(func, key))
            if key in func.__annotations__:
                raise TypeError("{0} got multiple values for annotation '{1}'".format(func, key))

            func.__annotations__[key] = value

        return func

    return wrap

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
