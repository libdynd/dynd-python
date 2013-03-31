Debugging DyND
==============

One of the simplest tools to get started debugging and
understanding how DyND works is the ``debug_repr()``
method which exists on most objects.

Here are a few examples to show what it prints.

Debug Printing NDObjects
-----------------------

.. code-block:: python

    >>> print(nd.ndobject(True).debug_repr())
    ------ ndobject
     address: 0000000007BBBC70
     refcount: 1
     dtype:
      pointer: 0000000000000001
      type: bool
     metadata:
      flags: 5 (read_access immutable )
     data:
       pointer: 0000000007BBBCA0
       reference: 0000000000000000 (embedded in ndobject memory)
    ------
    >>> print(nd.ndobject("testing").debug_repr())
    ------ ndobject
     address: 0000000007BBA6E0
     refcount: 1
     dtype:
      pointer: 0000000007BBBC70
      type: string<'ascii'>
     metadata:
      flags: 5 (read_access immutable )
      dtype-specific metadata:
       string metadata
        ------ memory_block at 0000000003F08510
         reference count: 1
         type: external
         object void pointer: 00000000027898A0
         free function: 000007FEEAC61974
        ------
     data:
       pointer: 0000000007BBA718
       reference: 0000000000000000 (embedded in ndobject memory)
    ------

    >>> print(nd.ndobject([1,2,3,4,5]).debug_repr())
    ------ ndobject
     address: 0000000003F0C7E0
     refcount: 1
     dtype:
      pointer: 0000000007BBB3D0
      type: strided_dim<int32>
     metadata:
      flags: 3 (read_access write_access )
      dtype-specific metadata:
       strided_dim metadata
        stride: 4
        size: 5
     data:
       pointer: 0000000003F0C820
       reference: 0000000000000000 (embedded in ndobject memory)
    ------

Debug Printing GFuncs
---------------------

This functionality was disabled during code refactoring,
and will resurface at some point in the future.

.. code-block:: python

    >>> nd.maximum.debug_dump()

    >>> nd.sum.debug_dump()

