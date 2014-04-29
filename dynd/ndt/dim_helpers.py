from __future__ import absolute_import, division, print_function

from dynd._pydynd import w_type, \
        make_var_dim, make_strided_dim, make_fixed_dim, make_cfixed_dim

__all__ = ['var', 'strided', 'fixed', 'cfixed']


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
            #  ndt.strided * 'int32'
            #  ndt.strided * int
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


class _Strided(_Dim):
    """
    Creates a strided dimension when combined with other types.

    Examples
    --------
    >>> ndt.strided * ndt.int32
    ndt.type('strided * int32')
    >>> ndt.fixed[5] * ndt.strided * ndt.float64
    ndt.type('5 * strided * float64')
    """
    __slots__ = []

    @property
    def dims(self):
        return (self,)

    def create(self, eltype):
        return make_strided_dim(eltype)

    def __repr__(self):
        return 'ndt.strided'


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
        if self.dim_size is not None:
            return (self,)
        else:
            raise TypeError('Need to specify ndt.fixed[dim_size],' +
                            ' not just ndt.fixed')

    def create(self, eltype):
        return make_fixed_dim(self.dim_size, eltype)

    def __getitem__(self, dim_size):
        return _Fixed(dim_size)

    def __repr__(self):
        if self.dim_size is not None:
            return 'ndt.fixed[%d]' % self.dim_size
        else:
            return 'ndt.fixed'


class _CFixed(_Dim):
    """
    Creates a cfixed dimension when combined with other types.

    Examples
    --------
    >>> ndt.cfixed[3] * ndt.int32
    ndt.type('cfixed[3] * int32')
    >>> ndt.fixed[5] * ndt.cfixed[2] * ndt.float64
    ndt.type('5 * cfixed[2] * float64')
    """
    __slots__ = ['dim_size']

    def __init__(self, dim_size = None):
        self.dim_size = dim_size

    @property
    def dims(self):
        if self.dim_size is not None:
            return (self,)
        else:
            raise TypeError('Need to specify ndt.cfixed[dim_size],' +
                            ' not just ndt.cfixed')

    def create(self, eltype):
        return make_cfixed_dim(self.dim_size, eltype)

    def __getitem__(self, dim_size):
        return _CFixed(dim_size)

    def __repr__(self):
        if self.dim_size is not None:
            return 'ndt.cfixed[%d]' % self.dim_size
        else:
            return 'ndt.cfixed'


var = _Var()
strided = _Strided()
fixed = _Fixed()
cfixed = _CFixed()
