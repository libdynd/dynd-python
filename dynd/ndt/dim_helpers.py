from __future__ import absolute_import, division, print_function

from dynd.ndt.type import make_var_dim, make_fixed_dim, make_fixed_dim_kind, type as w_type

__all__ = ['var', 'fixed']


class _Dim(object):
    __slots__ = []

    def __mul__(self, rhs):
        if isinstance(rhs, w_type):
            # Apply all the dimensions to get
            # produce a type
            for dim in reversed(self.dims):
                rhs = dim.create(rhs)
            return rhs
        elif isinstance(rhs, (str, type)):
            # Allow:
            #  ndt.fixed * 'int32'
            #  ndt.fixed * int
            rhs = w_type(rhs)
            for dim in reversed(self.dims):
                rhs = dim.create(rhs)
            return rhs
        elif isinstance(rhs, _Dim):
            # Combine the dimension fragments
            return _DimFragment(self.dims + rhs.dims)
        else:
            raise TypeError('Expected a dynd dimension or type, not %r' % rhs)

    def __pow__(self, count):
        return _DimFragment(self.dims * count)


class _DimFragment(_Dim):
    __slots__ = ['dims']

    def __init__(self, dims):
        self.dims = dims

    def __repr__(self):
        return ' * '.join(repr(dim) for dim in self.dims)


class _Var(_Dim):
    """
    Creates a var dimension when combined with other types.

    Examples
    --------
    >>> ndt.var * ndt.int32
    ndt.type('var * int32')
    >>> ndt.fixed[5] * ndt.var * ndt.float64
    ndt.type('5 * var * float64')
    """
    __slots__ = []

    @property
    def dims(self):
        return (self,)

    def create(self, eltype):
        return make_var_dim(eltype)

    def __repr__(self):
        return 'ndt.var'


class _Fixed(_Dim):
    """
    Creates a fixed dimension when combined with other types.

    Examples
    --------
    >>> ndt.fixed[3] * ndt.int32
    ndt.type('3 * int32')
    >>> ndt.fixed[5] * ndt.var * ndt.float64
    ndt.type('5 * var * float64')
    """
    __slots__ = ['dim_size']

    def __init__(self, dim_size = None):
        self.dim_size = dim_size

    @property
    def dims(self):
        return (self,)

    def create(self, eltype):
        if self.dim_size is None:
            return make_fixed_dim_kind(eltype)
        else:
            return make_fixed_dim(self.dim_size, eltype)

    def __getitem__(self, dim_size):
        return _Fixed(dim_size)

    def __repr__(self):
        if self.dim_size is not None:
            return 'ndt.fixed[%d]' % self.dim_size
        else:
            return 'ndt.fixed'


var = _Var()
fixed = _Fixed()