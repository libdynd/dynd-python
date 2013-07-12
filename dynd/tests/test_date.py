import sys
import unittest
from datetime import date
from dynd import nd, ndt

class TestDate(unittest.TestCase):
    def test_date_type_properties(self):
        self.assertEqual(type(ndt.date), ndt.type)
        self.assertEqual(str(ndt.date), 'date')
        self.assertEqual(ndt.date.data_size, 4)
        self.assertEqual(ndt.date.data_alignment, 4)
        self.assertEqual(ndt.date.canonical_type, ndt.date)

    def test_date_properties(self):
        a = nd.array(date(1955,3,13))
        self.assertEqual(str(a), '1955-03-13')
        self.assertEqual(nd.dtype_of(a.year.eval()), ndt.int32)
        self.assertEqual(nd.dtype_of(a.month.eval()), ndt.int32)
        self.assertEqual(nd.dtype_of(a.day.eval()), ndt.int32)
        self.assertEqual(nd.as_py(a.year), 1955)
        self.assertEqual(nd.as_py(a.month), 3)
        self.assertEqual(nd.as_py(a.day), 13)

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
                            [ndt.int32, ndt.int16, ndt.int16],
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

if __name__ == '__main__':
    unittest.main()
