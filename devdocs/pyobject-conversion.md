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

#### Simple cases

From existing tests, we have

```python

```

### `(array[Any]) -> pyobject`


