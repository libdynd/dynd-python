from cpython.ref cimport PyObject
from libc.stdint cimport intptr_t

from ..cpp.type cimport type as _type
from ..cpp.types.callable_type cimport make_callable
from ..cpp.func.callable cimport callable as _callable
from ..cpp.func.elwise cimport elwise as _elwise
from ..cpp.func.reduction cimport reduction as _reduction
from ..cpp.array cimport array as _array

from ..config cimport translate_exception
from ..wrapper cimport wrap, begin, end
from .callable cimport callable
from .. import ndt
from ..ndt.type cimport type, as_numba_type, from_numba_type

cdef extern from "arrfunc_from_pyfunc.hpp" namespace "pydynd::nd::functional":
    _callable _apply 'pydynd::nd::functional::apply'(object, object) except +translate_exception

cdef extern from "kernels/apply_jit_kernel.hpp" namespace "pydynd::nd::functional":
    _callable _apply_jit "pydynd::nd::functional::apply_jit"(const _type &tp, intptr_t)

    cdef cppclass jit_dispatcher:
        jit_dispatcher(object, object (*)(object, intptr_t, const _type *))

cdef extern from 'dynd/func/multidispatch.hpp' namespace 'dynd::nd::functional':
    _callable _multidispatch 'dynd::nd::functional::multidispatch'[T](_type, T, size_t) \
        except +translate_exception
    _callable _multidispatch 'dynd::nd::functional::multidispatch'[T](_type, T, T) \
        except +translate_exception

def _import_numba():
    try:
        import numba
    except ImportError:
        return False

    return True

cdef public object _jit(object func, intptr_t nsrc, const _type *src_tp):
    from llvmlite import ir

    CharType = ir.IntType(8)
    CharPointerType = CharType.as_pointer()
    Int32Type = ir.IntType(32)
    Int64Type = ir.IntType(64)

    def add_single_ir(ir_module):
        single = ir.Function(ir_module, ir.FunctionType(ir.VoidType(),
            [CharPointerType, CharPointerType.as_pointer()]),
            name = 'single')

        bb_entry = single.append_basic_block('entry')
        ir_builder = ir.IRBuilder(bb_entry)

        src = []
        for i, ir_type in enumerate(wrapped_func_ir_tp.args[-nsrc::]):
            src.append(ir_builder.load(ir_builder.bitcast(ir_builder.load(ir_builder.gep(single.args[1],
                [ir.Constant(Int32Type, i)])), ir_type.as_pointer())))

        status, dst = target_context.call_conv.call_function(ir_builder, wrapped_func,
            fndesc.restype, fndesc.argtypes, src)

        ir_builder.store(dst,
            ir_builder.bitcast(single.args[0], wrapped_func_ir_tp.args[0]))
        ir_builder.ret_void()

        return single

    # This is the Numba signature
    signature = tuple(as_numba_type(src_tp[i]) for i in range(nsrc))

    # Compile the function with Numba
    func.compile(signature)
    compile_res = func._compileinfos[signature]

    # Check if there is a corresponding return type in DyND
    cdef _type dst_tp = from_numba_type(compile_res.signature.return_type)

    # The following generates the wrapper function using LLVM IR
    fndesc = compile_res.fndesc
    target_context = compile_res.target_context
    library = target_context.jit_codegen().create_library(name = 'library')

    ir_module = library.create_ir_module(name = 'module')
#    memcpy = ir_module.declare_intrinsic('llvm.memcpy',
 #       [CharPointerType, CharPointerType, Int32Type])

    wrapped_func_ir_tp = target_context.call_conv.get_function_type(fndesc.restype,
        fndesc.argtypes)
    wrapped_func = ir_module.get_or_insert_function(wrapped_func_ir_tp,
        name = fndesc.llvm_func_name)

    single = add_single_ir(ir_module)

    library.add_ir_module(ir_module)
    library.finalize()

    return wrap(_apply_jit(make_callable(dst_tp, _array(src_tp, nsrc)),
            library.get_pointer_to_function('single')))

def apply(func = None, jit = _import_numba(), *args, **kwds):
    def make(tp, func):
        if jit:
            import numba
            return wrap(_multidispatch((<type> tp).v,
                jit_dispatcher(numba.jit(func, *args, **kwds), _jit), <size_t> 0))

        return wrap(_apply(func, tp))

    if func is None:
        return lambda func: make(ndt.callable(func), func)

    return make(ndt.callable(func), func)

def elwise(func):
    if not isinstance(func, callable):
        func = apply(func)

    return wrap(_elwise((<callable> func).v))

def reduction(child):
    if not isinstance(child, callable):
        child = apply(child)

    return wrap(_reduction((<callable> child).v))

def multidispatch(type tp, iterable = None):
    return wrap(_multidispatch(tp.v, begin[_callable](iterable),
        end[_callable](iterable)))
