Running The Python Tests
========================

The Python tests are written using Python's built in unittest module,
and can either be run by executing the scripts or with nose. To run
a test script directly, simply execute the test .py file:

    D:\Develop\dynamicndarray>python python\pydnd\tests\test_dtype.py
    ........
    ----------------------------------------------------------------------
    Ran 8 tests in 0.006s

    OK

To use nose, execute the following:

    D:\Develop\dnd\dynamicndarray>nosetests python\pydnd\tests
    ........................
    ----------------------------------------------------------------------
    Ran 24 tests in 0.119s

    OK

To generate Jenkins-compatible XML output, use
`nosetests --with-xunit --xunit-file=dynd_python_tests.xml python/pydnd/tests`.
