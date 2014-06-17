from __future__ import print_function, absolute_import
import unittest
from dynd import nd, ndt, _lowlevel

class TestEvalContext(unittest.TestCase):
    def test_basic_properties(self):
        ectx = nd.eval_context()
        # The default settings of an evaluation context
        self.assertEqual(ectx.errmode, 'fractional')
        self.assertEqual(ectx.cuda_device_errmode, 'nocheck'),
        self.assertEqual(ectx.date_parse_order, 'NoAmbig')
        self.assertEqual(ectx.century_window, 70)

    def test_modified_properties(self):
        ectx = nd.eval_context(errmode='overflow',
                               cuda_device_errmode='fractional',
                               date_parse_order='YMD',
                               century_window=1929)
        self.assertEqual(ectx.errmode, 'overflow')
        self.assertEqual(ectx.cuda_device_errmode, 'fractional'),
        self.assertEqual(ectx.date_parse_order, 'YMD')
        self.assertEqual(ectx.century_window, 1929)
        self.assertEqual(repr(ectx),
                         "nd.eval_context(errmode='overflow',\n" +
                         "                cuda_device_errmode='fractional',\n" +
                         "                date_parse_order='YMD',\n" +
                         "                century_window=1929)")

    def test_eval_errmode(self):
        a = nd.array(1.5).cast(ndt.int32)
        # Default error mode raises
        self.assertRaises(RuntimeError, a.eval)
        # But with an evaluation context with a 'nocheck' default error mode...
        self.assertEqual(nd.as_py(a.eval(ectx=nd.eval_context(errmode='nocheck'))),
                         1)

    def test_modify_default(self):
        try:
            # By default these should fail
            self.assertRaises(ValueError, nd.array, '123X', ndt.int32)
            self.assertRaises(ValueError, nd.array, '01/02/03', ndt.date)
            # If we disable error checking, the first will work
            nd.modify_default_eval_context(errmode='nocheck')
            self.assertEqual(nd.as_py(nd.array('123X', ndt.int32)), 123)
            self.assertRaises(ValueError, nd.array, '01/02/03', ndt.date)
            # If we set a default date parse error, the second will also work
            nd.modify_default_eval_context(date_parse_order='YMD')
            self.assertEqual(nd.as_py(nd.array('123X', ndt.int32)), 123)
            self.assertEqual(str(nd.array('01/02/03', ndt.date)), '2001-02-03')
            # Can change more than one setting at once
            nd.modify_default_eval_context(date_parse_order='DMY',
                                           errmode='overflow')
            self.assertRaises(ValueError, nd.array, '123X', ndt.int32)
            self.assertEqual(str(nd.array('01/02/03', ndt.date)), '2003-02-01')
            # Reset back to factory settings
            nd.modify_default_eval_context(reset=True)
            self.assertRaises(ValueError, nd.array, '123X', ndt.int32)
            self.assertRaises(ValueError, nd.array, '01/02/03', ndt.date)
        finally:
            nd.modify_default_eval_context(reset=True)

if __name__ == '__main__':
    unittest.main()
