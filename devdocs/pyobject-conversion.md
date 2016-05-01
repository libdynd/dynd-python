# PyObject to DyND Array and Back

For seamless interoperability with Python, DyND needs extremely solid conversion between Python objects and nd::arrays. What exactly that means isn't completely obvious, as NumPy's experience shows, in having some excellent behavior but sometimes doing unexpected things.

## Requirements

* Assignment needs to be pluggable
  * Dynamically registered DyND types should be able to specify Python type conversions
  * Newly defined Python types should be able to specify conversions
* Automatic type deductions should be unsurprising, or fail if no reasonable choice (e.g. avoid copying NumPy 'object' array behaviour)
* Trivial cases with Python native types should be very fast (e.g. `nd.array(3.14)`, `nd.array([1,2,3,4])`)
* Nested generators and iterators should work intuitively. This means conversion with deduction should make only one pass through the PyObject dimensional structure, because processing an iterator consumes it.

## The Types Framing the Discussion

All DyND conversions are defined in terms of a source type and destination type, e.g. `(float32) -> int64`. In DyND, the `nd::array` object can be placed inside another `nd::array` via the `array[Any]` type. This allows signatures for general conversions to/from DyND to be expressed the same way as specific conversions for types like `float32`. The Python bindings define a type `pyobject` which holds a single reference to a Python object, i.e. a `PyObject *` pointer in the case of the CPython interpreter. 

Combining these give us the two general conversion signatures, `(pyobject) -> array[Any]` and `(array[Any]) -> pyobject`.

## Example Conversions

Instead of beginning this with general rules for conversion, let's go through a bunch of specific examples and use our intuition to define what we think the conversion should be. Then we can take the collection of examples and produce a system which matches it. Iterating the system through finding examples that behave unintuitively, adding them to this list, and tweaking the system to account for them should hopefully converge to a good end result.

### `(pyobject) -> array[Any]`

#### Scalar cases

The following is from existing tests as well as some additional examples

```python
# Python object => DyND type
True                                           => "bool"
10                                             => "int32"
-2200000000                                    => "int64"
5.125                                          => "float64"
5.125 - 2.5j                                   => "complex[float64]"
'abcdef'                                       => "string"
u'abcdef'                                      => "string"
b'abcdef'                                      => "bytes"          # On Python 3
```

#### Array cases

```python
[]                                             => "0 * int32"      # Current choice for empty dynamic list
[[], [], []]                                   => "3 * 0 * int32"
[1, 2, 3]                                      => "3 * int32"
[True, False]                                  => "2 * bool"
[1, True]                                      => "2 * int32"      # Current integer behaviour
[10000000000, 1, False]                        => "3 * int64"
[10000000000, 3.25, 2, False]                  => "4 * float64"
[3.25j, 3.25, 1, 2, True]                      => "5 * complex[float64]"
[str(x) + 'test' for x in range(10)]           => "10 * string"
[u'test', 'test2']                             => "2 * string"
[b'x'*x for x in range(10)]                    => "10 * bytes"    # On Python 3
[[True, 2, 3], [4, 5, 6.5], [1, 2, 3]]         => "3 * 3 * float64"
[[1], [2, 3, 4], [5, 6]]                       => "3 * var * int32"
[[True, False], [False, 2, 3], [-10000000000], [True, 10, 3.125, 5.5j]] => "4 * var * complex[float64]"
[[], [False, 2, 3]]                            => "2 * var * int32"
[[], [[]], [[[1, 3]]]]                         => "3 * var * var * 2 * int32"
```

#### Iterator protocol cases

Generators and other iterators can present tricky cases.

```python
(x for x in [])                                => "0 * int32" # Equivalent to []
iter([...]) for all list cases above           => "..."       # Same as the list version but as iterator
```

#### Error cases

The following should throw errors (or we should at least think hard about how "smart" DyND should be).

```python
[[1], [[2]]]       # The 1 and 2 values imply different numbers of dimensions
[1, "test"]        # Implicitly converting numbers to strings not allowed
[b"test", "test"]  # (Python 3) Implicitly converting bytes to strings or vice versa is not allowed

```

#### Questions

* What should the DyND type be for an empty dynamic list? This question applies across all such contexts, and should be the same for JSON parsing, Python object conversion, etc. Current choice is `0 * int32`.
* What should the deduced integer type be?
  * `int32` if it fits, `int64` if not, `int128` if still not, `bigint` if still not.
  * `int64` if it fits, `int128` if not, `bigint` if still not
  * `bigint` (SSO-based arbitrary-sized integer)
* Do we like Python's implicit bool to int conversion? Is it an option for us to raise an error for it instead?

### `(array[Any]) -> pyobject`

