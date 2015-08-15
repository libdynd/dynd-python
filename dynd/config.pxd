cdef:
    void translate_exception()
    void set_broadcast_exception(object)

    const void *dynd_get_lowlevel_api()
    const void *dynd_get_py_lowlevel_api()
