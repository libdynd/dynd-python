Debugging DyND
==============

One of the simplest tools to get started debugging and
understanding how DyND works is the ``nd.debug_repr(a)``
method which exists on most objects.

Here are a few examples to show what it prints.

Debug Printing ND.Array
-----------------------

.. code-block:: python

    >>> nd.debug_repr(nd.array(True))
    ------ array
     address: 000000000281BE10
     refcount: 1
     type:
      pointer: 0000000000000001
      type: bool
     metadata:
      flags: 5 (read_access immutable )
     data:
       pointer: 000000000281BE40
       reference: 0000000000000000 (embedded in array memory)
    ------

    >>> nd.debug_repr(nd.array("testing"))
    ------ array
     address: 000000000281F7A0
     refcount: 1
     type:
      pointer: 000000000281BE10
      type: string
     metadata:
      flags: 5 (read_access immutable )
      dtype-specific metadata:
       string metadata
        ------ NULL memory block
     data:
       pointer: 000000000281F7D8
       reference: 0000000000000000 (embedded in array memory)
    ------

    >>> nd.debug_repr(nd.array([1,2,3,4,5]))
    ------ array
     address: 000000000281F7A0
     refcount: 1
     type:
      pointer: 00000000028212A0
      type: strided_dim<int32>
     metadata:
      flags: 3 (read_access write_access )
      dtype-specific metadata:
       strided_dim metadata
        stride: 4
        size: 5
     data:
       pointer: 000000000281F7E0
       reference: 0000000000000000 (embedded in array memory)
    ------


Debug Printing GFuncs
---------------------

This functionality was disabled during code refactoring,
and will resurface at some point in the future.

.. code-block:: python

    >>> nd.maximum.debug_dump()

    >>> nd.sum.debug_dump()

