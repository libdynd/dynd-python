Running The Python Tests
========================

The Python tests are written using Python's built in unittest module,
and can either be run by executing the scripts or with nose. To run
a test script directly, simply execute the test .py file:

    D:\Develop\dynd-python>python dynd\tests\test_dtype.py
    ...........
    ----------------------------------------------------------------------
    Ran 11 tests in 0.003s

    OK

To execute the full test suite, run the following (not from the root project
directory):

    D:\Develop>python -c "import dynd; dynd.test()"
    Running unit tests for the DyND Python bindings
    Python version: 3.3.2 |Continuum Analytics, Inc.| (default, May 17 2013, 11:32:27) [MSC v.1500 64 bit (AMD64)]
    Python prefix: C:\Anaconda\envs\py33
    DyND-Python module: C:\Anaconda\envs\py33\lib\site-packages\dynd
    DyND-Python version: 0.3.1-8-gd0620eb-dirty
    DyND-Python git sha1: d0620ebad73a9ef840e40c9649ba97bff9444e0c
    LibDyND version: 0.3.1-2-g25d6d4e
    LibDyND git sha1: 25d6d4ef6ba4b1213359b0da80897c04c8d7474a
    NumPy version: 1.7.1
    ..........................................................................................................
    ----------------------------------------------------------------------
    Ran 106 tests in 0.348s

    OK

To generate Jenkins-compatible XML output, control the verbosity of the
output, etc, you may pass some extra parameters to the test function.

    D:\Develop>python
    Python 3.3.2 |Continuum Analytics, Inc.| (default, May 17 2013, 11:32:27) [MSC v.1500 64 bit (AMD64)] on win32
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import dynd
    >>> help(dynd.test)
    Help on function test in module dynd:

    test(verbosity=1, xunitfile=None, exit=False)
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
