from dynd.wrapper cimport wrap, begin, end
from .. import ndt
from .callable cimport _callable, callable
from .array cimport _array

from dynd.ndt.type cimport type, int32_type_id, int64_type_id, float32_type_id, float64_type_id, make_callable

def apply(tp_or_func, func = None):
    def make(tp, func):
        return wrap(_apply(func, tp))

    if func is None:
        if isinstance(tp_or_func, ndt.type):
            return lambda func: make(tp_or_func, func)

        return make(ndt.callable(tp_or_func), tp_or_func)

    return make(tp_or_func, func)

def elwise(func):
    if not isinstance(func, callable):
        func = apply(func)

    return wrap(_elwise((<callable> func).v))

def multidispatch(type tp, iterable = None):
    return wrap(_multidispatch(tp.v, begin[_callable](iterable),
        end[_callable](iterable)))

def apply_ptr(tp, ptr):
    return wrap(numba_helper((<type> tp).v, <intptr_t> ptr))

import numba

as_numba_tp = {}
as_numba_tp[int32_type_id] = numba.int32
as_numba_tp[int64_type_id] = numba.int64
as_numba_tp[float64_type_id] = numba.float32
as_numba_tp[float64_type_id] = numba.float64

as_dynd_tp = {}
as_dynd_tp[numba.int32] = ndt.type(int32_type_id)
as_dynd_tp[numba.int64] = ndt.type(int64_type_id)
as_dynd_tp[numba.float32] = ndt.type(float32_type_id)
as_dynd_tp[numba.float64] = ndt.type(float64_type_id)


from numba import types, compiler, njit, cgutils

cdef public object jit_func(object f, intptr_t nsrc, const _type *src_tp):
    from llvmlite import ir

    res = f.compile(tuple(as_numba_tp[src_tp[i].get_type_id()] for i in range(nsrc)))
    sig = f.signatures[0]

    compile_res = f._compileinfos[sig]

    cdef _type dst_tp = (<type> as_dynd_tp[compile_res.signature.return_type]).v
 #   dst_tp = _type(as_numba_tp[compile_res.signature.return_type])
#    print dst_tp
 #   if  not in as_numba_tp.values():
  #      print 'error' # need to sort this out
   #     raise Exception('no numba type')

    fndesc = compile_res.fndesc
    target_context = compile_res.target_context

    library = target_context.jit_codegen().create_library('dynd_example_library')
    mod = library.create_ir_module(name='dynd_example')

    inner_func_type = target_context.call_conv.get_function_type(fndesc.restype,
        fndesc.argtypes)
    inner_func = mod.get_or_insert_function(inner_func_type,
                                            name=fndesc.llvm_func_name)

    CharType = ir.IntType(8)

    single = ir.Function(mod, ir.FunctionType(ir.VoidType(),
        [CharType.as_pointer(), CharType.as_pointer().as_pointer()]),
        name = 'single')

    bb_entry = single.append_basic_block('entry')
    irbuilder = ir.IRBuilder(bb_entry)

    src = []
    for i, ir_type in enumerate(inner_func_type.args[-nsrc::]):
        src.append(irbuilder.load(irbuilder.bitcast(irbuilder.load(irbuilder.gep(single.args[1],
            [ir.Constant(ir.IntType(32), i)])), ir_type.as_pointer())))

    status, dst = target_context.call_conv.call_function(irbuilder, inner_func,
        fndesc.restype, fndesc.argtypes, src)

    irbuilder.store(dst,
        irbuilder.bitcast(single.args[0], inner_func_type.args[0]))
    irbuilder.ret_void()

    library.add_ir_module(mod)
    library.finalize()

    return apply_ptr(wrap(make_callable(dst_tp, _array(src_tp, nsrc))),
            library.get_pointer_to_function('single'))

def apply_numba(tp, f):
    return wrap(_multidispatch2((<type> tp).v, jit_dispatcher(f, jit_func), <size_t> 0))