def get_test_module(m):
    exec('import %s as tst' % m)
    return tst
