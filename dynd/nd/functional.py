import atexit, distutils, distutils.core, imp, os, os.path, shutil, sys, tempfile

from dynd import include_dirs
import dynd

def inline(statement, header = ''):
  source = '''
    #include <dynd/func/apply.hpp>
    #include <pydynd/array_functions.hpp>

    using namespace std;
    using namespace pydynd;

    {0}

    static PyObject *make(PyObject *self, PyObject *args)
    {{
      return wrap_array({1});
    }}

    static PyMethodDef InlineMethods[] = {{
        {{"make", make, METH_NOARGS, "Return the arrfunc."}},
        {{NULL, NULL, 0, NULL}}
    }};

#if PY_MAJOR_VERSION >= 3
    #define PY_MODINIT(NAME) PyMODINIT_FUNC PyInit_##NAME()
#else
    #define PY_MODINIT(NAME) PyMODINIT_FUNC init##NAME()
#endif

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef InlineModule = {{
      PyModuleDef_HEAD_INIT,
      "inline",
      "Wraps a C++ function.",
      -1,
      InlineMethods,
      NULL,
      NULL,
      NULL,
      NULL,
    }};
#endif

    PY_MODINIT(inline)
    {{
#if PY_MAJOR_VERSION >= 3
      return PyModule_Create(&InlineModule);
#else
      Py_InitModule3("inline", InlineMethods, "Wraps a C++ function.");
#endif
    }}
  '''

  tempdir = tempfile.mkdtemp()
  atexit.register(shutil.rmtree, tempdir)

  srcfile = open(os.path.join(tempdir, 'inline.cpp'), 'w')
  srcfile.write(source.format(header, statement))
  srcfile.close()

  if (sys.platform == 'darwin'):
    extra_link_args = []
  else:
    if (sys.version_info >= (3,0)):
      import sysconfig
      extra_link_args = [os.path.join(os.path.dirname(dynd.__file__), '_pydynd' + sysconfig.get_config_var('SO'))]
    else:
      extra_link_args = [os.path.join(os.path.dirname(dynd.__file__), '_pydynd.so')]

  ext = distutils.core.Extension('inline', [srcfile.name], extra_compile_args = ['-std=c++11'],
    include_dirs = include_dirs, language = 'c++', libraries = ['dynd'], 
    extra_link_args = extra_link_args)
  distutils.core.setup(name = ext.name, ext_modules = [ext], script_name = 'functional.py', script_args = ['--quiet',
    'build_ext', '--build-temp', os.path.join(tempdir, 'build'), '--build-lib', tempdir])

  modfile, moddir, moddescr = imp.find_module(ext.name, [tempdir])
  mod = imp.load_module(ext.name, modfile, moddir, moddescr)
  modfile.close()

  return mod.make()