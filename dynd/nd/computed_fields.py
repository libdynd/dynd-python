__all__ = ['add_computed_fields']

from dynd._pydynd import as_py, as_numpy, w_type, \
                w_array as array, make_cstruct, \
                elwise_map, extract_dtype, type_of, \
                dtype_of, ndim_of

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
        import datetime
        import numpy
        import scipy
        self.glbl = {}
        self.glbl.update(datetime.__dict__)
        self.glbl.update(numpy.__dict__)
        self.glbl.update(scipy.__dict__)
        self.glbl['as_py'] = as_py
        self.glbl['as_numpy'] = as_numpy

    def __call__(self, dst, src):
        # Loop element by element
        for dst_itm, src_itm in zip(dst, src):
            # Put all the src fields in a locals dict
            lcl = {}
            for i, name in enumerate(self.src_field_names):
                s = getattr(src_itm, name).eval()
                if ndim_of(s) > 0 or dtype_of(s).kind == 'struct':
                    # For types which NumPy doesn't support, leave
                    # them as DyND arrays
                    try:
                        s = as_numpy(s, allow_copy=True)
                    except TypeError:
                        pass
                else:
                    s = as_py(s)

                lcl[str(name)] = s

            # Evaluate all the field exprs
            for i, expr in enumerate(self.dst_field_expr):
                v = eval(expr, self.glbl, lcl)
                dst_itm[i] = v

def add_computed_fields(n, fields, rm_fields=[], fnname=None):
    """
    Adds one or more new fields to a struct array,
    using nd.elwise_map to create the deferred object.

    Each field_expr should be a string or bit of code
    that can be evaluated with an 'eval' call. It is called
    with numpy/scipy in the globals, and the input
    fields in the locals.

    Parameters
    ----------
    n : dynd array
        This should have a struct data dtype. The
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
    ndt.type('strided_dim<expr<cstruct<float64 r, float64 theta>, op0=cstruct<float64 x, float64 y>, expr=topolar(op0)>>')
    >>> y.eval()
    nd.array([[2, 0], [2, -1.5708], [5.83095, 1.03038], [5.65685, 0.785398]], strided_dim<cstruct<float64 r, float64 theta>>)
    >>> x[0] = (-100, 0)
    >>> y[0].eval()
    nd.array([100, 3.14159], cstruct<float64 r, float64 theta>)
    """
    n = array(n)
    udt = dtype_of(n).value_type
    if udt.kind != 'struct':
        raise ValueError("parameter 'n' must have kind 'struct'")

    # The field names and types of the input struct
    field_names = as_py(udt.field_names)
    field_types = as_py(udt.field_types)
    # Put the new field names in a dict as well
    new_field_dict = {}
    for fn, ft, fe in fields:
        new_field_dict[fn] = w_type(ft)

    # Create the output struct type and corresponding expressions
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

    result_udt = make_cstruct(new_field_types, new_field_names)
    fieldexpr = FieldExpr(new_field_expr, field_names, fnname)

    return elwise_map([n], fieldexpr, result_udt)

def make_computed_fields(n, replace_ndim, fields, fnname=None):
    """
    Creates a new struct type, with fields computed based
    on the input fields. Leaves the requested number of
    array dimensions in place, so the result has fewer
    than the input if positive.

    Each field_expr should be a string or bit of code
    that can be evaluated with an 'eval' call. It is called
    with numpy/scipy in the globals, and the input
    fields in the locals.

    Parameters
    ----------
    n : dynd array
        This should have a struct data type. The
        result will be a view of this data.
    replace_ndim : integer
        The number of array dimensions to leave in the
        input going to the fields. For example if the
        input has shape (3,4,2) and replace_ndim is 1,
        the result will have shape (3,4), and each operand
        provided to the field expression will have shape (2).
    fields : list of (field_name, field_type, field_expr)
        These are the fields which are created in the output.
        No fields are retained from the input.
    fnname : string, optional
        The function name, which affects how the resulting
        deferred expression's type is printed.

    Examples
    --------
    >>> from dynd import nd, ndt
    >>> a = nd.array([
    ...  ('A', 1, 2), ('A', 3, 4),
    ...  ('B', 1.5, 2.5), ('A', 0.5, 9),
    ...  ('C', 1, 5), ('B', 2, 2)],
    ...  dtype='{cat: string; x: float32; y: float32}')
    >>> gb = nd.groupby(a, a.cat)
    >>> gb.groups
    nd.array(["A", "B", "C"], strided_dim<string>)
    >>> b = nd.make_computed_fields(gb.eval(), 1,
    ...                 fields=[('sum_x', ndt.float32, 'sum(x)'),
    ...                         ('mean_y', ndt.float32, 'mean(y)'),
    ...                         ('max_x', ndt.float32, 'max(x)'),
    ...                         ('max_y', ndt.float32, 'max(y)'),
    ...                         ('min_y', ndt.float32, 'min(y)')])
    >>> from pprint import pprint
    >>> pprint(nd.as_py(b))
    [{u'max_x': 3.0, u'max_y': 9.0, u'mean_y': 5.0, u'min_y': 2.0, u'sum_x': 4.5},
     {u'max_x': 2.0, u'max_y': 2.5, u'mean_y': 2.25, u'min_y': 2.0, u'sum_x': 3.5},
     {u'max_x': 1.0, u'max_y': 5.0, u'mean_y': 5.0, u'min_y': 5.0, u'sum_x': 1.0}]
    """
    n = array(n)
    udt = dtype_of(n).value_type
    if udt.kind != 'struct':
        raise ValueError("parameter 'n' must have kind 'struct'")

    # The field names and types of the input struct
    field_names = as_py(udt.field_names)
    field_types = as_py(udt.field_types)

    # Create the output struct type and corresponding expressions
    new_field_names = []
    new_field_types = []
    new_field_expr = []
    for fn, ft, fe in fields:
        new_field_names.append(fn)
        new_field_types.append(ft)
        new_field_expr.append(fe)

    result_udt = make_cstruct(new_field_types, new_field_names)
    src_udt = extract_dtype(type_of(n), replace_ndim)
    fieldexpr = FieldExpr(new_field_expr, field_names, fnname)

    return elwise_map([n], fieldexpr, result_udt, [src_udt])
