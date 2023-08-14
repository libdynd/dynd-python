import unittest
# import nd to make sure the types are propagated out of the registry.
from dynd import ndt, nd

class TestTypeBasics(unittest.TestCase):
    def test_type_repr(self):
        roundtrip = [
          "fixed_string[10, 'utf16']",
          "() -> int32",
          "(int32) -> int32",
          "(int32, float64) -> int32",
          "(..., scale: float64, ...) -> int32",
          "(int32, ..., scale: float64, color: float64, ...) -> int32",
          "(int32, float32, ..., scale: float64, color: float64, ...) -> int32"
        ]

        for s in roundtrip:
            self.assertEqual(repr(ndt.type(s)), "ndt.type(" + repr(s) + ")")

if __name__ == '__main__':
    unittest.main(verbosity=2)
