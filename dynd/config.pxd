cdef:
    void translate_exception()
    void set_broadcast_exception(object)

cdef api object DyND_PyExc_BroadcastError
