import sys
import unittest
import ctypes
from datetime import date, time
from dynd import nd, ndt

class TestDate(unittest.TestCase):
    def test_date_type_properties(self):
        self.assertEqual(type(ndt.date), ndt.type)
        self.assertEqual(str(ndt.date), 'date')
        self.assertEqual(ndt.date.data_size, 4)
        self.assertEqual(ndt.date.data_alignment, 4)
        self.assertEqual(ndt.date.canonical_type, ndt.date)

    def test_date_properties(self):
        a = nd.array(date(1955, 3, 13))
        self.assertEqual(str(a), '1955-03-13')
        self.assertEqual(nd.dtype_of(a.year.eval()), ndt.int32)
        self.assertEqual(nd.dtype_of(a.month.eval()), ndt.int32)
        self.assertEqual(nd.dtype_of(a.day.eval()), ndt.int32)
        self.assertEqual(nd.as_py(a.year), 1955)
        self.assertEqual(nd.as_py(a.month), 3)
        self.assertEqual(nd.as_py(a.day), 13)
        self.assertEqual(nd.as_py(a), date(1955, 3, 13))

    def test_struct_casting(self):
        a = nd.array([date(1912,3,4), date(2002,1,30)])
        # cast from date to struct
        s = a.ucast(ndt.make_cstruct([ndt.int64, ndt.int16, ndt.int8],
                                        ['year', 'month', 'day']))
        s = s.eval()
        self.assertEqual(nd.as_py(s.year), [1912, 2002])
        self.assertEqual(nd.as_py(s.month), [3, 1])
        self.assertEqual(nd.as_py(s.day), [4, 30])
        # cast from struct back to date
        d = s.ucast(ndt.date)
        d = d.eval()
        self.assertEqual(nd.as_py(d), [date(1912,3,4), date(2002,1,30)])

    def test_struct_function(self):
        a = nd.array(date(1955,3,13))
        s = a.to_struct().eval()
        self.assertEqual(nd.dtype_of(s),
                        ndt.make_cstruct(
                            [ndt.int16, ndt.int8, ndt.int8],
                            ['year', 'month', 'day']))
        self.assertEqual(nd.as_py(s.year), 1955)
        self.assertEqual(nd.as_py(s.month), 3)
        self.assertEqual(nd.as_py(s.day), 13)

    def test_strftime(self):
        a = nd.array(date(1955,3,13))
        self.assertEqual(nd.as_py(a.strftime('%Y')), '1955')
        self.assertEqual(nd.as_py(a.strftime('%m/%d/%y')), '03/13/55')
        self.assertEqual(nd.as_py(a.strftime('%Y and %j')), '1955 and 072')

        a = nd.array([date(1931,12,12), date(2013,5,14), date(2012,12,25)])
        self.assertEqual(nd.as_py(a.strftime('%Y-%m-%d %j %U %w %W')),
                        ['1931-12-12 346 49 6 49', '2013-05-14 134 19 2 19', '2012-12-25 360 52 2 52'])

    def test_weekday(self):
        self.assertEqual(nd.as_py(nd.array(date(1955,3,13)).weekday()), 6)
        self.assertEqual(nd.as_py(nd.array(date(2002,12,4)).weekday()), 2)

    def test_replace(self):
        a = nd.array(date(1955,3,13))
        self.assertEqual(nd.as_py(a.replace(2013)), date(2013,3,13))
        self.assertEqual(nd.as_py(a.replace(2012,12)), date(2012,12,13))
        self.assertEqual(nd.as_py(a.replace(2012,12,15)), date(2012,12,15))
        # Custom extension, allow -1 indexing from the end for months and days
        self.assertEqual(nd.as_py(a.replace(month=7)), date(1955,7,13))
        self.assertEqual(nd.as_py(a.replace(day=-1,month=7)), date(1955,7,31))
        self.assertEqual(nd.as_py(a.replace(month=2,day=-1)), date(1955,2,28))
        self.assertEqual(nd.as_py(a.replace(month=2,day=-1,year=2000)), date(2000,2,29))

    def test_date_parse(self):
        # By default, don't allow ambiguous year interpretations
        self.assertRaises(ValueError, nd.array('01/02/03').cast('date').eval)
        self.assertEqual(nd.as_py(nd.array('01/02/03').cast('date').eval(
                                ectx=nd.eval_context(date_parse_order='YMD'))),
                         date(2001, 2, 3))
        self.assertEqual(nd.as_py(nd.array('01/02/03').cast('date').eval(
                                ectx=nd.eval_context(date_parse_order='MDY'))),
                         date(2003, 1, 2))
        self.assertEqual(nd.as_py(nd.array('01/02/03').cast('date').eval(
                                ectx=nd.eval_context(date_parse_order='DMY'))),
                         date(2003, 2, 1))
        # Can also change the two year handling
        # century_window of 0 means don't allow
        self.assertRaises(ValueError, nd.array('01/02/03').cast('date').eval,
                          ectx=nd.eval_context(date_parse_order='YMD',
                                               century_window=0))
        # century_window of 10 means a sliding window starting 10 years ago
        self.assertEqual(nd.as_py(nd.array('01/02/03').cast('date').eval(
                                ectx=nd.eval_context(date_parse_order='YMD',
                                                     century_window=10))),
                         date(2101, 2, 3))
        # century_window of 1850 means a fixed window starting at 1850
        self.assertEqual(nd.as_py(nd.array('01/02/03').cast('date').eval(
                                ectx=nd.eval_context(date_parse_order='YMD',
                                                     century_window=1850))),
                         date(1901, 2, 3))

    def test_json_date_parse(self):
        a = nd.parse_json('var * date', '["2012-03-17", "1922-12-30"]')
        self.assertEqual(nd.as_py(a), [date(2012, 3, 17), date(1922, 12, 30)])
        self.assertRaises(ValueError, nd.parse_json, 'var * date',
                          '["2012-03-17T17:00:15-0600", "1922-12-30 Thursday"]')
        a = nd.parse_json('var * date',
                          '["2012-06-17T17:00:15-0600", "1921-12-30 Thursday"]',
                          ectx=nd.eval_context(errmode='nocheck'))
        self.assertEqual(nd.as_py(a), [date(2012, 6, 17), date(1921, 12, 30)])

class TestTime(unittest.TestCase):
    def test_time_type_properties(self):
        self.assertEqual(type(ndt.time), ndt.type)
        self.assertEqual(str(ndt.time), 'time')
        self.assertEqual(ndt.time.data_size, 8)
        self.assertEqual(ndt.time.data_alignment,
                         ctypes.alignment(ctypes.c_int64))
        self.assertEqual(ndt.time.canonical_type, ndt.time)

    def test_time_properties(self):
        a = nd.array(time(14, 25, 59, 123456))
        self.assertEqual(str(a), '14:25:59.123456')
        self.assertEqual(nd.dtype_of(a.hour.eval()), ndt.int32)
        self.assertEqual(nd.dtype_of(a.minute.eval()), ndt.int32)
        self.assertEqual(nd.dtype_of(a.second.eval()), ndt.int32)
        self.assertEqual(nd.dtype_of(a.microsecond.eval()), ndt.int32)
        self.assertEqual(nd.dtype_of(a.tick.eval()), ndt.int32)
        self.assertEqual(nd.as_py(a.hour), 14)
        self.assertEqual(nd.as_py(a.minute), 25)
        self.assertEqual(nd.as_py(a.second), 59)
        self.assertEqual(nd.as_py(a.microsecond), 123456)
        self.assertEqual(nd.as_py(a.tick), 1234560)
        self.assertEqual(nd.as_py(a), time(14, 25, 59, 123456))

    def test_struct_casting(self):
        a = nd.array([time(13, 25, 8, 765432), time(23, 52)])
        # cast from time to struct
        s = a.ucast(ndt.make_cstruct([ndt.int64, ndt.int16, ndt.int8, ndt.int32],
                                        ['hour', 'minute', 'second', 'tick']))
        s = s.eval()
        self.assertEqual(nd.as_py(s.hour), [13, 23])
        self.assertEqual(nd.as_py(s.minute), [25, 52])
        self.assertEqual(nd.as_py(s.second), [8, 0])
        self.assertEqual(nd.as_py(s.tick), [7654320, 0])
        # cast from struct back to time
        t = s.ucast(ndt.time)
        t = t.eval()
        self.assertEqual(nd.as_py(t), [time(13, 25, 8, 765432), time(23, 52)])

    def test_struct_function(self):
        a = nd.array(time(13, 25, 8, 765432))
        s = a.to_struct().eval()
        self.assertEqual(nd.dtype_of(s),
                        ndt.make_cstruct(
                            [ndt.int8, ndt.int8, ndt.int8, ndt.int32],
                            ['hour', 'minute', 'second', 'tick']))
        self.assertEqual(nd.as_py(s.hour), 13)
        self.assertEqual(nd.as_py(s.minute), 25)
        self.assertEqual(nd.as_py(s.second), 8)
        self.assertEqual(nd.as_py(s.tick), 7654320)


if __name__ == '__main__':
    unittest.main(verbosity=2)
