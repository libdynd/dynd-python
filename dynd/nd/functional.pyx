from dynd.wrapper cimport wrap, begin, end
from .. import ndt
from .callable cimport _callable, callable
from .array cimport _array

from dynd.ndt.type cimport type, make_callable
from dynd.ndt.type cimport as_numba_type, from_numba_type

# This takes a Python func passed through the jit decorator, and returns
# a DyND callable object
cdef public object jit_func(object func, intptr_t nsrc, const _type *src_tp):
    from llvmlite import ir

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

    wrapped_func_ir_tp = target_context.call_conv.get_function_type(fndesc.restype,
        fndesc.argtypes)
    wrapped_func = ir_module.get_or_insert_function(wrapped_func_ir_tp,
        name = fndesc.llvm_func_name)

    CharType = ir.IntType(8)

    # DyND needs a function pointer of type void (char *dst, char *const *src)
    single = ir.Function(ir_module, ir.FunctionType(ir.VoidType(),
        [CharType.as_pointer(), CharType.as_pointer().as_pointer()]),
        name = 'single')

    bb_entry = single.append_basic_block('entry')
    irbuilder = ir.IRBuilder(bb_entry)


    # For each argument i, do the following:
    # 1) Use GEP to get the char ** corresponding to src[i]
    # 2) Load the char * from that char **
    # 3) Cast it to its appropriate type expected by Numba.
    # 4) Load its value

    src = []
    for i, ir_type in enumerate(wrapped_func_ir_tp.args[-nsrc::]):
        src.append(irbuilder.load(irbuilder.bitcast(irbuilder.load(irbuilder.gep(single.args[1],
            [ir.Constant(ir.IntType(32), i)])), ir_type.as_pointer())))

    # Call the wrapped Numba function pointer with the values of src
    status, dst = target_context.call_conv.call_function(irbuilder, wrapped_func,
        fndesc.restype, fndesc.argtypes, src)

    # Store the returned value from the Numba function
    irbuilder.store(dst,
        irbuilder.bitcast(single.args[0], wrapped_func_ir_tp.args[0]))
    irbuilder.ret_void()

    library.add_ir_module(ir_module)
    library.finalize()

    return wrap(_apply_jit(make_callable(dst_tp, _array(src_tp, nsrc)),
            library.get_pointer_to_function('single')))

def apply(tp_or_func, func = None):
    def make(tp, func):
        try:
            from numba import jit
            return wrap(_multidispatch((<type> tp).v,
                jit_dispatcher(jit(func, nopython = True), jit_func), <size_t> 0))
        except ImportError:
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