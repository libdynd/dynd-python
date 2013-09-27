from __future__ import absolute_import

import operator

def _validate_squeeze_index(i, sz):
    try:
        i = operator.index(i)
    except TypeError:
        raise TypeError('nd.squeeze() requires an int or '
                    + 'tuple of ints for axis parameter')
    if i >= 0:
        if i >= sz:
            raise IndexError('nd.squeeze() axis %d is out of range' % i)
    else:
        if i < -sz:
            raise IndexError('nd.squeeze() axis %d is out of range' % i)
        else:
            i += sz
    return i

def squeeze(a, axis=None):
    """Removes size-one dimensions from the beginning and end of the shape.
    If `axis` is provided, removes the dimensions specified in the tuple.

    Parameters
    ----------
    a : nd.array
        The array to be squeezed.
    axis : int or tuple of int, optional
        Specifies which exact dimensions to remove. The dimensions specified
        must have size one.
    """
    s = a.shape
    ssz = len(s)
    ix = [slice(None)]*ssz
    if axis is not None:
        if isinstance(axis, tuple):
            axis = [_validate_squeeze_index(x, ssz) for x in axis]
        else:
            axis = [_validate_squeeze_index(axis, ssz)]
        for x in axis:
            if s[x] != 1:
                raise IndexError(('nd.squeeze() requested axis %d ' +
                        'has shape %d, not 1 as required') % (x, s[x]))
            ix[x] = 0
    else:
        # Construct a list of indexer objects which trim off the
        # beginning and end
        for i in range(ssz):
            if s[i] == 1:
                ix[i] = 0
            else:
                break
        for i in range(ssz-1, -1, -1):
            if s[i] == 1:
                ix[i] = 0
            else:
                break
    ix = tuple(ix)
    return a[ix]
