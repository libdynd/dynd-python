import os, sys
import unittest
import importlib

def discover():
    all_suites = []
    for f in os.listdir(os.path.dirname(__file__)):
        if f.startswith('test') and f.endswith('.py'):
            name = '.' + os.path.splitext(f)[0]
            m = importlib.import_module(name, package=__name__)
            suite = unittest.defaultTestLoader.loadTestsFromModule(m)
            all_suites.append(suite)
    return all_suites

def run(stream=sys.stderr, verbosity=2):
    all_suites = discover()
    runner = unittest.TextTestRunner(stream=stream, verbosity=verbosity)
    return runner.run(unittest.TestSuite(all_suites))
