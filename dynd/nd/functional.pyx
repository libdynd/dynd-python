from libc.stdint cimport intptr_t
from libcpp.vector cimport vector
from cpython cimport PyObject

from ..cpp.callable cimport callable as _callable
from ..cpp.type cimport type as _type
from ..cpp.types.callable_type cimport callable_type as _callable_type
from ..cpp.functional cimport elwise as _elwise
from ..cpp.functional cimport reduction as _reduction
from ..cpp.array cimport array as _array
from ..cpp.type cimport make_type

from ..config cimport translate_exception
from .array cimport _functional_apply as _apply
from .callable cimport callable, wrap, dynd_nd_callable_to_cpp
from ..ndt.type cimport type, as_numba_type, from_numba_type, as_cpp_type

cdef extern from 'dynd/functional.hpp' namespace 'dynd::nd::functional':
    _callable _dispatch 'dynd::nd::functional::dispatch'[T](_type, T) \
        except +translate_exception
    _callable _multidispatch 'dynd::nd::functional::multidispatch'[T](_type, T, T) \
        except +translate_exception

cdef extern from 'dynd/callable.hpp' namespace 'dynd::nd':
    _callable _make_callable 'dynd::nd::make_callable'[T](_type, object, ...) except +translate_exception

cdef extern from "callables/apply_jit_callable.hpp" namespace "pydynd::nd::functional":
    _callable _apply_jit "pydynd::nd::functional::apply_jit"(const _type &tp, intptr_t)

    cdef cppclass apply_jit_dispatch_callable:
        apply_jit_dispatch_callable(object, object (*)(object, intptr_t, const _type *))

def _import_numba():
    try:
        import numba
    except ImportError:
        return False

    return False

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
    compile_res = func.overloads[signature]

    # Check if there is a corresponding return type in DyND
    cdef _type dst_tp = from_numba_type(compile_res.signature.return_type)

    # The following generates the wrapper function using LLVM IR
    fndesc = compile_res.fndesc
    target_context = compile_res.target_context
    library = target_context.codegen().create_library(name = 'library')

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

    cdef vector[_type] src_tp_copy
    for i in range(nsrc):
        src_tp_copy.push_back(src_tp[i])

    return wrap(_apply_jit(make_type[_callable_type](dst_tp, src_tp_copy),
            library.get_pointer_to_function('single')))

def apply(func = None, jit = _import_numba(), *args, **kwds):
    from .. import ndt
    def make(type tp, func):
        if jit:
            import numba
            return wrap(_make_callable[apply_jit_dispatch_callable]((<type> tp).v,
                <object> numba.jit(func, *args, **kwds), _jit))

        return wrap(_apply(tp.v, func))

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

"""
def multidispatch(type tp, iterable = None):
    cdef vector[_callable] v
    if iterable is not None:
        for c in iterable:
            v.push_back(dynd_nd_callable_to_cpp(c))
    return wrap(_multidispatch(as_cpp_type(tp), v.begin(), v.end()))
"""
