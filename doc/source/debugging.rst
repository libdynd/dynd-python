Debugging Blaze-Local
=====================

One of the simplest tools to get started debugging and
understanding how blaze-local works is the ``debug_dump()``
method which exists on most objects.

Here are a few examples to show what it prints.

Debug Printing NDArrays
-----------------------

.. code-block:: python

    >>> nd.ndarray(True).debug_dump()
    ------ ndarray
     ("immutable_scalar",
      dtype: bool
      ndim: 0
      shape: ()
      node category: strided_array_node_category
      access flags: read immutable 
      data: 01
      value: true
     )
    ------

    >>> nd.ndarray("testing").debug_dump()
    ------ ndarray
     ("scalar",
      dtype: string<ascii>
      ndim: 0
      shape: ()
      node category: strided_array_node_category
      access flags: read immutable
      data: 08370803000000000f37080300000000
      blockref memory block
      ------ memory_block at 00000000028995E0
       reference count: 1
       type: external
      ------
     )
    ------

    >>> nd.ndarray([1,2,3,4,5]).debug_dump()
    ------ ndarray
     ("strided_array",
      dtype: int32
      ndim: 1
      shape: (5)
      node category: strided_array_node_category
      access flags: read write
      strides: (4)
      originptr: 000000000293F680
      memoryblock owning the data:
      ------ memory_block at 000000000293F670
       reference count: 1
       type: fixed_size_pod
       no blockrefs
      ------
     )
    ------

    >>> (nd.ndarray([1,2,3,4,5]) + 1).debug_dump()
    ------ ndarray
     ("elementwise_binary_kernel",
      dtype: int32
      ndim: 1
      shape: (5)
      node category: elwise_node_category
      access flags: read
      nop: 2
      operand 0:
       ("strided_array",
        dtype: int32
        ndim: 1
        shape: (5)
        node category: strided_array_node_category
        access flags: read write
        strides: (4)
        originptr: 000000000293F680
        memoryblock owning the data:
        ------ memory_block at 000000000293F670
         reference count: 1
         type: fixed_size_pod
         no blockrefs
        ------
       )
      operand 1:
       ("immutable_scalar",
        dtype: int32
        ndim: 0
        shape: ()
        node category: strided_array_node_category
        access flags: read immutable
        data: 01000000
        value: 1
       )
     )
    ------

Debug Printing GFuncs
---------------------

.. code-block:: python

    >>> nd.maximum.debug_dump()
    ------ elwise_gfunc
    name: maximum
    kernel count: 6
    kernel 0
       int32 (int32, int32)
    binary aux data: 0000000002939A10
    kernel 1
       int64 (int64, int64)
    binary aux data: 0000000002939A50
    kernel 2
       uint32 (uint32, uint32)
    binary aux data: 0000000002939A90
    kernel 3
       uint64 (uint64, uint64)
    binary aux data: 0000000002939AD0
    kernel 4
       float32 (float32, float32)
    binary aux data: 0000000002939B10
    kernel 5
       float64 (float64, float64)
    binary aux data: 0000000002939B50
    ------

    >>> nd.sum.debug_dump()
    ------ elwise_reduce_gfunc
    name: sum
    kernel count: 6
    kernel 0
     signature: int32 (int32)
     associative: true
     commutative: true
     left associative kernel aux data: 0000000001E6E9E0
     reduction identity:
      ------ ndarray
       ("immutable_builtin_scalar",
        dtype: int32
        ndim: 0
        shape: ()
        node category: strided_array_node_category
        access flags: read immutable
        data: 00000000
        value: 0
       )
      ------
    kernel 1
     signature: int64 (int64)
     associative: true
     commutative: true
     left associative kernel aux data: 0000000001E6EA20
     reduction identity:
      ------ ndarray
       ("immutable_scalar",
        dtype: int64
        ndim: 0
        shape: ()
        node category: strided_array_node_category
        access flags: read immutable
        data: 0000000000000000
        value: 0
       )
      ------
    kernel 2
     signature: uint32 (uint32)
     associative: true
     commutative: true
     left associative kernel aux data: 0000000001E6EA60
     reduction identity:
      ------ ndarray
       ("immutable_scalar",
        dtype: uint32
        ndim: 0
        shape: ()
        node category: strided_array_node_category
        access flags: read immutable
        data: 00000000
        value: 0
       )
      ------
    kernel 3
     signature: uint64 (uint64)
     associative: true
     commutative: true
     left associative kernel aux data: 0000000001E6EAA0
     reduction identity:
      ------ ndarray
       ("immutable_scalar",
        dtype: uint64
        ndim: 0
        shape: ()
        node category: strided_array_node_category
        access flags: read immutable
        data: 0000000000000000
        value: 0
       )
      ------
    kernel 4
     signature: float32 (float32)
     associative: true
     commutative: true
     left associative kernel aux data: 0000000001E6EAE0
     reduction identity:
      ------ ndarray
       ("immutable_scalar",
        dtype: float32
        ndim: 0
        shape: ()
        node category: strided_array_node_category
        access flags: read immutable
        data: 00000000
        value: 0
       )
      ------
    kernel 5
     signature: float64 (float64)
     associative: true
     commutative: true
     left associative kernel aux data: 0000000001E6EB20
     reduction identity:
      ------ ndarray
       ("immutable_scalar",
        dtype: float64
        ndim: 0
        shape: ()
        node category: strided_array_node_category
        access flags: read immutable
        data: 0000000000000000
        value: 0
       )
      ------
    ------

