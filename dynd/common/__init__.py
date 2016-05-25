from ..config import _dynd_version_string as __libdynd_version__, \
                     _dynd_python_version_string as __version__, \
                     _dynd_git_sha1 as __libdynd_git_sha1__, \
                     _dynd_python_git_sha1 as __git_sha1__, \
                     load

__all__ = [
    '__libdynd_version__', '__version__', '__libdynd_git_sha1__', '__git_sha1__',
    'annotate', 'test', 'load'
]

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
    import unittest
    import dynd.ndt.test as ndt_test
    try:
        import dynd.nd.test as nd_test
    except ImportError:
        nd_test = None

    print('Running unit tests for the DyND Python bindings')
    print('Python version: %s' % sys.version)
    print('Python prefix: %s' % sys.prefix)
    print('DyND-Python module: %s' % os.path.dirname(os.path.dirname(__file__)))
    print('DyND-Python version: %s' % __version__)
    print('DyND-Python git sha1: %s' % __git_sha1__)
    print('LibDyND version: %s' % __libdynd_version__)
    print('LibDyND git sha1: %s' % __libdynd_git_sha1__)
    print('NumPy version: %s' % numpy.__version__)
    sys.stdout.flush()

    if xunitfile is None:
        s1 = ndt_test.discover()
        s2 = nd_test.discover() if nd_test else []
        runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=verbosity)
        result = runner.run(unittest.TestSuite(s1 + s2))
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
            if 'test' in dirs:
                testsdir = os.path.join(root, 'test')
                argv.append(testsdir)
                print('Test dir: %s' % testsdir[len(rootdir)+1:])
        # Ask nose to do its thing
        return nose.main(argv=argv, exit=exit)
