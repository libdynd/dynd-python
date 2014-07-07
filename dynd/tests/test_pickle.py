import sys
import unittest
from pickle import loads, dumps
from dynd import nd, ndt

class TestPickle(unittest.TestCase):
    def test_pickle_types(self):
        self.assertEqual(nd.array, loads(dumps(nd.array)))
        self.assertEqual(nd.eval_context, loads(dumps(nd.eval_context)))
        self.assertEqual(nd.arrfunc, loads(dumps(nd.arrfunc)))
        self.assertEqual(ndt.type, loads(dumps(ndt.type)))
