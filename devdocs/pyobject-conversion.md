# PyObject to DyND Array and Back

For seamless interoperability with Python, DyND needs extremely solid conversion between Python objects and nd::arrays. What exactly that means isn't completely obvious, as can be seen in NumPy's choices which often are great, and sometimes surprising.

## Requirements

* Assignment needs to be pluggable
  * Dynamically registered DyND types should be able to specify Python type conversions
  * Newly defined Python types should be able to specify conversions
* Automatic type deductions should be unsurprising, or fail if no reasonable choice (e.g. avoid copying NumPy 'object' array behaviour)
* Trivial cases with Python native types should be very fast (e.g. `nd.array(3.14)`, `nd.array([1,2,3,4])`)
* Nested generators and iterators should work intuitively. This means conversion with deduction should make only one pass through the PyObject dimensional structure, because processing an iterator consumes it.
