from __future__ import absolute_import, division, print_function

from dynd._pydynd import w_type, \
        make_var_dim, make_strided_dim, make_fixed_dim, make_cfixed_dim

__all__ = ['DimCollector', 'var', 'strided', 'fixed', 'cfixed']


class DimHelper(object):
    __slots__ = []

    def __mul__(self, rhs):
        if isinstance(rhs, w_type):
            # Apply all the dimensions to get
            # produce a type
            for dim in reversed(self.dims):
                rhs = dim.create(rhs)
            return rhs
        elif isinstance(rhs, DimHelper):
            # Combine the dimension fragments
            return DimCollector(*(self.dims + rhs.dims))
        else:
            raise TypeError('Expected a dynd dimension or type, not %r' % rhs)


class DimCollector(DimHelper):
    __slots__ = ['dims']

    def __init__(self, *dims):
        self.dims = dims

    def __repr__(self):
        return 'ndt.DimCollector(%s)' % ', '.join(repr(dim)
                                                  for dim in self.dims)

class VarHelper(DimHelper):
    """
    Creates a var dimension when combined with other types.

    Examples
    --------

    >>> ndt.var * ndt.int32
    """
    __slots__ = []

    @property
    def dims(self):
        return [self]

    def create(self, eltype):
        return make_var_dim(eltype)

    def __repr__(self):
        return 'ndt.var'


class StridedHelper(DimHelper):
    __slots__ = []

    @property
    def dims(self):
        return (self,)

    def create(self, eltype):
        return make_strided_dim(eltype)

    def __repr__(self):
        return 'ndt.strided'

    def __pow__(self, count):
        return DimCollector(*((self,) * count))


class FixedHelper(DimHelper):
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
        return FixedHelper(dim_size)

    def __repr__(self):
        if self.dim_size is not None:
            return 'ndt.fixed[%d]' % self.dim_size
        else:
            return 'ndt.fixed'


class CFixedHelper(DimHelper):
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
        return CFixedHelper(dim_size)

    def __repr__(self):
        if self.dim_size is not None:
            return 'ndt.cfixed[%d]' % self.dim_size
        else:
            return 'ndt.cfixed'


var = VarHelper()
strided = StridedHelper()
fixed = FixedHelper()
cfixed = CFixedHelper()
