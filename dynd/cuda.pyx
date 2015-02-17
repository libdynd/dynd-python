from cython cimport address

cdef class event(object):
    def __cinit__(self):
        throw_if_not_cuda_success(cudaEventCreate(address(self._event)))

    def __dealloc__(self):
        throw_if_not_cuda_success(cudaEventDestroy(self._event))

    def record(self):
        throw_if_not_cuda_success(cudaEventRecord(self._event))

    def synchronize(self):
        throw_if_not_cuda_success(cudaEventSynchronize(self._event))

    def elapsed_time(self, event other):
        cdef float ms
        throw_if_not_cuda_success(cudaEventElapsedTime(address(ms), other._event, self._event))
        return ms