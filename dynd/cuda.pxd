include "config.pxi"

IF DYND_CUDA:
    cdef extern from "cuda_runtime_api.h":
        cdef cppclass cudaError_t:
            pass

        cdef cppclass cudaEvent_t:
            pass

        cudaError_t cudaEventCreate(cudaEvent_t *event)    
        cudaError_t cudaEventDestroy(cudaEvent_t event)
        cudaError_t cudaEventRecord(cudaEvent_t event)
        cudaError_t cudaEventSynchronize(cudaEvent_t event)
        cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t stop)

    cdef extern from "dynd/exceptions.hpp" namespace "dynd":
        void cuda_throw_if_not_success(cudaError_t error)

    cdef class event(object):
        cdef cudaEvent_t _event