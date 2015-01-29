import tempfile
import os

from distutils.core import Extension, setup

import imp
import atexit
import shutil

def inline(statement, header = ''):
  source = '''
    #include <dynd/func/apply.hpp>

    #include "/Users/irwin/Repositories/dynd-python/include/array_functions.hpp"

    using namespace std;
    using namespace dynd;

    {}

    static PyObject *make(PyObject *self, PyObject *args)
    {{
      return pydynd::wrap_array({});
    }}

    static PyMethodDef InlineMethods[] = {{
        {{"make", make, METH_NOARGS, "Return the arrfunc."}},
        {{NULL, NULL, 0, NULL}}
    }};

    PyMODINIT_FUNC initinline()
    {{
      Py_InitModule("inline", InlineMethods);
    }}
  '''

  tempdir = tempfile.mkdtemp()
  atexit.register(shutil.rmtree, tempdir)

  srcfile = open(os.path.join(tempdir, 'inline.cpp'), 'w')
  srcfile.write(source.format(header, statement))
  srcfile.close()

  ext = Extension('inline', [srcfile.name], language = 'c++', extra_compile_args = ['-std=c++11'])
  setup(name = ext.name, ext_modules = [ext], script_name = 'functional.py', script_args = ['--quiet',
    'build_ext', '--build-temp', os.path.join(tempdir, 'build'), '--build-lib', tempdir])

  modfile, moddir, moddescr = imp.find_module(ext.name, [tempdir])
  mod = imp.load_module(ext.name, modfile, moddir, moddescr)
  return mod.make()