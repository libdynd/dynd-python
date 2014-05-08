import sys
import unittest
from dynd import nd, ndt

class TestArrayAsPy(unittest.TestCase):
    def test_struct_or_tuple(self):
        a = nd.array((3, "testing", 1.5), type='{x:int, y:string, z:real}')
        self.assertEqual(nd.as_py(a), {'x': 3, 'y': "testing", 'z': 1.5})
        self.assertEqual(nd.as_py(a, tuple=True), (3, "testing", 1.5))
        a = nd.array([(1, 1.5), (2, 3.5)], dtype='{x:int, y:real}')
        self.assertEqual(nd.as_py(a), [{'x': 1, 'y': 1.5}, {'x': 2, 'y': 3.5}])
        self.assertEqual(nd.as_py(a, tuple=True), [(1, 1.5), (2, 3.5)])

        # Slightly bigger example
        data = {
            "type": "ImageCollection",
            "images": [{
                   "Width":  800,
                    "Height": 600,
                    "Title":  "View from 15th Floor",
                    "Thumbnail": {
                        "Url":    "http://www.example.com/image/481989943",
                        "Height": 125,
                        "Width":  100
                    },
                    "IDs": [116, 943, 234, 38793]
                }]
        }
        ordered = (u'ImageCollection',
                [(800, 600, u'View from 15th Floor',
                    (u'http://www.example.com/image/481989943', 125, 100),
                    [116, 943, 234, 38793]),])

        tp = ndt.type("""{
              type: string,
              images: var * {
                    Width: int16,
                    Height: int16,
                    Title: string,
                    Thumbnail: {
                        Url: string,
                        Height: int16,
                        Width: int16,
                    },
                    IDs: var * int32,
                }
            }
            """)
        a = nd.array(data, type=tp)
        self.assertEqual(nd.as_py(a), data)
        self.assertEqual(nd.as_py(a, tuple=True), ordered)
