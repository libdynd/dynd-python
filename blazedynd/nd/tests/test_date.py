import sys
import unittest
from datetime import date
from blazedynd import nd

class TestDate(unittest.TestCase):
    def test_date_dtype_properties(self):
        self.assertEqual(type(nd.dt.date), nd.dtype)
        self.assertEqual(str(nd.dt.date), 'date')
        self.assertEqual(nd.dt.date.element_size, 4)
        self.assertEqual(nd.dt.date.alignment, 4)
        self.assertEqual(nd.dt.date.canonical_dtype, nd.dt.date)
        self.assertEqual(nd.dt.date.unit, 'day')

    def test_date_properties(self):
        a = nd.ndobject(date(1955,3,13))
        self.assertEqual(str(a), '1955-03-13')
        self.assertEqual(a.year.vals().dtype, nd.dt.int32)
        self.assertEqual(a.month.vals().dtype, nd.dt.int32)
        self.assertEqual(a.day.vals().dtype, nd.dt.int32)
        self.assertEqual(a.year.as_py(), 1955)
        self.assertEqual(a.month.as_py(), 3)
        self.assertEqual(a.day.as_py(), 13)

    def test_struct_casting(self):
        a = nd.ndobject([date(1912,3,4), date(2002,1,30)])
        # cast from date to struct
        s = a.cast_scalars(nd.dt.make_fixedstruct_dtype([nd.dt.int64, nd.dt.int16, nd.dt.int8],
                                        ['year', 'month', 'day']))
        s = s.vals()
        self.assertEqual(s.year.as_py(), [1912, 2002])
        self.assertEqual(s.month.as_py(), [3, 1])
        self.assertEqual(s.day.as_py(), [4, 30])
        # cast from struct back to date
        d = s.cast_udtype(nd.dt.date)
        d = d.vals()
        self.assertEqual(d.as_py(), [date(1912,3,4), date(2002,1,30)])

    def test_struct_function(self):
        a = nd.ndobject(date(1955,3,13))
        s = a.to_struct().vals()
        self.assertEqual(s.dtype,
                        nd.dt.make_fixedstruct_dtype([nd.dt.int32, nd.dt.int8, nd.dt.int8],
                                        ['year', 'month', 'day']))
        self.assertEqual(s.year.as_py(), 1955)
        self.assertEqual(s.month.as_py(), 3)
        self.assertEqual(s.day.as_py(), 13)

    def test_strftime(self):
        a = nd.ndobject(date(1955,3,13))
        self.assertEqual(a.strftime('%Y').as_py(), '1955')
        self.assertEqual(a.strftime('%m/%d/%y').as_py(), '03/13/55')
        self.assertEqual(a.strftime('%Y and %j').as_py(), '1955 and 072')

        a = nd.ndobject([date(1931,12,12), date(2013,5,14), date(2012,12,25)])
        self.assertEqual(a.strftime('%Y-%m-%d %j %U %w %W').as_py(),
                        ['1931-12-12 346 49 6 49', '2013-05-14 134 19 2 19', '2012-12-25 360 52 2 52'])
 
    def test_weekday(self):
        self.assertEqual(nd.ndobject(date(1955,3,13)).weekday().as_py(), 6)
        self.assertEqual(nd.ndobject(date(2002,12,04)).weekday().as_py(), 2)

    def test_replace(self):
        a = nd.ndobject(date(1955,3,13))
        self.assertEqual(a.replace(2013).as_py(), date(2013,3,13))
        self.assertEqual(a.replace(2012,12).as_py(), date(2012,12,13))
        self.assertEqual(a.replace(2012,12,15).as_py(), date(2012,12,15))
        # Custom extension, allow -1 indexing from the end for months and days
        self.assertEqual(a.replace(month=7).as_py(), date(1955,7,13))
        self.assertEqual(a.replace(day=-1,month=7).as_py(), date(1955,7,31))
        self.assertEqual(a.replace(month=2,day=-1).as_py(), date(1955,2,28))
        self.assertEqual(a.replace(month=2,day=-1,year=2000).as_py(), date(2000,2,29))

if __name__ == '__main__':
    unittest.main()
