import atexit, distutils, distutils.core, imp, os, os.path, shutil, tempfile

from dynd import include_dirs

def inline(statement, header = ''):
  source = '''
    #include <dynd/func/apply.hpp>
    #include <pydynd/array_functions.hpp>

    using namespace std;
    using namespace pydynd;

    {}

    static PyObject *make(PyObject *self, PyObject *args)
    {{
      return wrap_array({});
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

  ext = distutils.core.Extension('inline', [srcfile.name], language = 'c++', extra_compile_args = ['-std=c++11'], include_dirs = include_dirs)
  distutils.core.setup(name = ext.name, ext_modules = [ext], script_name = 'functional.py', script_args = ['--quiet',
    'build_ext', '--build-temp', os.path.join(tempdir, 'build'), '--build-lib', tempdir])

  modfile, moddir, moddescr = imp.find_module(ext.name, [tempdir])
  mod = imp.load_module(ext.name, modfile, moddir, moddescr)
  return mod.make()