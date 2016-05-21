def get_tst_module(m):
    l = dict(locals())
    exec('from . import %s as tst' % m, globals(), l)
    return l['tst']
