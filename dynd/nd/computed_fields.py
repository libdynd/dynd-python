__all__ = ['add_computed_fields']

from dynd._pydynd import as_py, as_numpy, w_dtype as dtype, \
                w_ndobject as ndobject, make_fixedstruct_dtype, \
                elwise_map

class FieldExpr:
    def __init__(self, dst_field_expr, src_field_names, fnname):
        if fnname is None:
            self.__name__ = 'computed_field_expr'
        else:
            self.__name__ = fnname
        self.dst_field_expr = dst_field_expr
        self.src_field_names = src_field_names
        # Create a globals dict containing numpy and scipy
        # for the expressions to use
        import numpy
        import scipy
        self.glbl = {}
        self.glbl.update(numpy.__dict__)
        self.glbl.update(scipy.__dict__)

    def __call__(self, dst, src):
        # Create a locals dict with all the fields
        lcl = {}
        for i, name in enumerate(self.src_field_names):
            lcl[str(name)] = as_numpy(src[:, i])
        
        # Evaluate all the field exprs
        for i, expr in enumerate(self.dst_field_expr):
            dst[:, i] = eval(expr, self.glbl, lcl)

def add_computed_fields(n, fields, rm_fields=[], fnname=None):
    """
    Adds one or more new fields to a struct ndobject,
    using nd.elwise_map to create the deferred object.
    
    Each field_expr should be a string or bit of code
    that can be evaluated with an 'eval' call. It is called
    with numpy/scipy in the globals, and the input
    fields in the locals.
    
    Parameters
    ----------
    n : ndobject
        This should have a uniform struct dtype. The
        result will be a view of this data.
    fields : list of (field_name, field_type, field_expr)
        These are the fields which are added to 'n'.
    rm_fields : list of string, optional
        For fields that are in the input, but have no expression,
        this removes them from the output struct instead of
        keeping the value.
    fnname : string, optional
        The function name, which affects how the resulting
        deferred expression's dtype is printed.

    Examples
    --------
    >>> from dynd import nd, ndt
    >>> import numpy as np

    >>> x = np.array([(2, 0), (0, -2), (3, 5), (4, 4)],
    ...         dtype=[('x', np.float64), ('y', np.float64)])
    >>> y = nd.add_computed_fields(x,
    ...         fields=[('r', np.float64, 'sqrt(x*x + y*y)'),
    ...                 ('theta', np.float64, 'arctan2(y, x)')],
    ...         rm_fields=['x', 'y'],
    ...         fnname='topolar')
    >>> y.dtype
    nd.dtype('strided_dim<expr<fixedstruct<float64 r, float64 theta>, op0=fixedstruct<float64 x, float64 y>, expr=topolar(op0)>>')
    >>> y.eval()
    nd.ndobject([[2, 0], [2, -1.5708], [5.83095, 1.03038], [5.65685, 0.785398]], strided_dim<fixedstruct<float64 r, float64 theta>>)
    >>> x[0] = (-100, 0)
    >>> y[0].eval()
    nd.ndobject([100, 3.14159], fixedstruct<float64 r, float64 theta>)
    """
    n = ndobject(n)
    udt = n.udtype
    if udt.kind != 'struct':
        raise ValueError("parameter 'n' must have kind 'struct'")

    # The field names and types of the input struct
    field_names = as_py(udt.field_names)
    field_types = as_py(udt.field_types)
    # Put the new field names in a dict as well
    new_field_dict = {}
    for fn, ft, fe in fields:
        new_field_dict[fn] = dtype(ft)

    # Create the output struct dtype and corresponding expressions
    new_field_names = []
    new_field_types = []
    new_field_expr = []
    for fn, ft in zip(field_names, field_types):
        if fn not in new_field_dict and fn not in rm_fields:
            new_field_names.append(fn)
            new_field_types.append(ft)
            new_field_expr.append(fn)
    for fn, ft, fe in fields:
        new_field_names.append(fn)
        new_field_types.append(ft)
        new_field_expr.append(fe)

    result_udt = make_fixedstruct_dtype(new_field_types, new_field_names)
    fieldexpr = FieldExpr(new_field_expr, field_names, fnname)

    return elwise_map([n], fieldexpr, result_udt)
