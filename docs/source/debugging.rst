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
     address: 0000000000415380
     refcount: 1
     type:
      pointer: 0000000000000001
      type: bool
     arrmeta:
      flags: 5 (read_access immutable )
     data:
       pointer: 00000000004153B0
       reference: 0000000000000000 (embedded in array memory)
    ------

    >>> nd.debug_repr(nd.array("testing"))
    ------ array
     address: 00000000003FAE90
     refcount: 1
     type:
      pointer: 0000000000415380
      type: string
     arrmeta:
      flags: 5 (read_access immutable )
      type-specific arrmeta:
       string arrmeta
        ------ NULL memory block
     data:
       pointer: 00000000003FAEC8
       reference: 0000000000000000 (embedded in array memory)
    ------

    >>> nd.debug_repr(nd.array([1,2,3,4,5]))
    ------ array
     address: 00000000003FAE90
     refcount: 1
     type:
      pointer: 000007FEE194E8F0
      type: strided * int32
     arrmeta:
      flags: 5 (read_access immutable )
      type-specific arrmeta:
       strided_dim arrmeta
        stride: 4
        size: 5
     data:
       pointer: 00000000003FAED0
       reference: 0000000000000000 (embedded in array memory)
    ------


Debug Printing GFuncs
---------------------

This functionality was disabled during code refactoring,
and will resurface at some point in the future.

.. code-block:: python

    >>> nd.maximum.debug_dump()

    >>> nd.sum.debug_dump()

