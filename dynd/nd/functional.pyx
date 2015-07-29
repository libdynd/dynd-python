from dynd.wrapper cimport wrap, begin, end
from .. import ndt
from .callable cimport _callable, callable

from dynd.ndt.type cimport type

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

dynd_to_numba = {}
dynd_to_numba[ndt.type('int32').type_id] = numba.int32
dynd_to_numba[ndt.type('int64').type_id] = numba.int64

def apply_numba(tp, f):
    import llvmlite.ir as ll
    import llvmlite.binding as llvm

    from numba import types, compiler, njit, cgutils

    print tp.pos_types

    res = f.compile(tuple(types.int32 for i in range(2)))
    sig = f.signatures[0]
    cres = f._compileinfos[sig]

    fndesc = cres.fndesc
    library = cres.library
    target = cres.target_context

    codegen = target.jit_codegen()
    wrapper_library = codegen.create_library('dynd_example_library')
    mod = wrapper_library.create_ir_module(name='dynd_example')

    inner_func_type = target.call_conv.get_function_type(fndesc.restype,
                                                     fndesc.argtypes)

    inner_func = mod.get_or_insert_function(inner_func_type,
                                            name=fndesc.llvm_func_name)

    CharType = ll.IntType(8)

    single = ll.Function(mod, ll.FunctionType(ll.VoidType(),
        [CharType.as_pointer(), CharType.as_pointer().as_pointer()]),
        name = 'single')

    bb_entry = single.append_basic_block('entry')
    irbuilder = ll.IRBuilder(bb_entry)

    dst_tp = ll.IntType(64)
    src_tp = [ll.IntType(32), ll.IntType(32)]

    src = []
    for i in range(2):
        src.append(irbuilder.load(irbuilder.bitcast(irbuilder.load(irbuilder.gep(single.args[1],
            [ll.Constant(ll.IntType(64), i)])), src_tp[i].as_pointer())))


    status, dst = target.call_conv.call_function(irbuilder, inner_func,
                                                    fndesc.restype,
                                                    fndesc.argtypes,
                                                    src)
    
    irbuilder.store(dst,
        irbuilder.bitcast(single.args[0], dst_tp.as_pointer()))
    irbuilder.ret_void()

    wrapper_library.add_ir_module(mod)
    wrapper_library.finalize()

    return apply_ptr(tp, wrapper_library.get_pointer_to_function('single'))