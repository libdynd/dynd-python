from cython cimport address

include "config.pxi"

IF DYND_CUDA:
    cdef class event(object):
        def __cinit__(self):
            cuda_throw_if_not_success(cudaEventCreate(address(self._event)))

        def __dealloc__(self):
            cuda_throw_if_not_success(cudaEventDestroy(self._event))

        def record(self):
            cuda_throw_if_not_success(cudaEventRecord(self._event))

        def synchronize(self):
            cuda_throw_if_not_success(cudaEventSynchronize(self._event))

        def elapsed_time(self, event other):
            cdef float ms
            cuda_throw_if_not_success(cudaEventElapsedTime(address(ms), other._event, self._event))
            return ms