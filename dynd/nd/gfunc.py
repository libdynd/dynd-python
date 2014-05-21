__all__ = ['elwise', 'mean1d']

#, 'codegen_cache', 'default_cgcache', 'elwise_reduce']

from dynd._pydynd import w_elwise_gfunc as elwise, \
        w_elwise_reduce_gfunc as elwise_reduce

#        default_cgcache as default_cgcache, \
#        w_codegen_cache as codegen_cache, \

from .._lowlevel import make_builtin_mean1d_arrfunc

mean1d = make_builtin_mean1d_arrfunc('float64', -1)
