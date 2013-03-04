def get_tst_module(m):
    exec('import %s as tst' % m)
    return tst
