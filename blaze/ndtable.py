"""
NDtables are distributed
Attributes:
   * nd -- number of dimensions === number of independent indexing objects
   * children --- a mapping of names or sub-indexes to ndtables
   * name --- this is a unique URL for the ndtable or empty string if local
   * indexes --- an nd-sequence of index-objects for each dimension. 
"""

############################
# Index Objects
############################
class Index(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

class Auto(Index):
    pass

class Concrete(Index):
    values = []

class Mask(Index):
    mask = []

class Function(Index):
    func = None

class FuncIndex(Function):
    pass

class FuncCallBack(Function):
    pass

# A Leaf Table is a local ndtable that is a leaf node
#  It is a collection of local index arrays
class LeafTable(object):
    pass

# An ExpressionTable is an NDTable that is a calculation graph
class ExpressionTable(object):
    pass

# A LocalTable is a collection of typed-bytes
class LocalTable(object):
    pass

class NDTable(object):
    nd = 1
    children = {}
    name = ''
    indexes = [Auto()]
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

