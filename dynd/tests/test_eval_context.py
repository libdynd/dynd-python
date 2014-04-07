from __future__ import print_function, absolute_import
import unittest
from dynd import nd, ndt, _lowlevel

class TestEvalContext(unittest.TestCase):
    def test_basic_properties(self):
        ectx = nd.eval_context()
        # The default settings of an evaluation context
        self.assertEqual(ectx.default_errmode, 'fractional')
        self.assertEqual(ectx.default_cuda_device_errmode, 'none'),
        self.assertEqual(ectx.date_parse_order, 'NoAmbig')
        self.assertEqual(ectx.century_window, 70)

    def test_modified_properties(self):
        ectx = nd.eval_context(default_errmode='overflow',
                               default_cuda_device_errmode='fractional',
                               date_parse_order='YMD',
                               century_window=1929)
        self.assertEqual(ectx.default_errmode, 'overflow')
        self.assertEqual(ectx.default_cuda_device_errmode, 'fractional'),
        self.assertEqual(ectx.date_parse_order, 'YMD')
        self.assertEqual(ectx.century_window, 1929)
        self.assertEqual(repr(ectx),
                         "nd.eval_context(default_errmode='overflow',\n" +
                         "                default_cuda_device_errmode='fractional',\n" +
                         "                date_parse_order='YMD',\n" +
                         "                century_window=1929)")

    def test_eval_errmode(self):
        a = nd.array(1.5).cast(ndt.int32)
        # Default error mode raises
        self.assertRaises(RuntimeError, a.eval)
        # But with an evaluation context with a 'none' default error mode...
        self.assertEqual(nd.as_py(a.eval(ectx=nd.eval_context(default_errmode='none'))),
                         1)

if __name__ == '__main__':
    unittest.main()
