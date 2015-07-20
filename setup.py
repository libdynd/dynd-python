from distutils.command.build_ext import build_ext
from distutils import sysconfig
#from distutils.core import setup
from setuptools import setup, Extension

import os, sys
from os import chdir, getcwd
from os.path import abspath, dirname

import re

# Check if we're running 64-bit Python
import struct
is_64_bit = struct.calcsize('@P') == 8

class cmake_build_ext(build_ext):
  description = "Build the C-extension for dynd-python with CMake"

  def get_ext_path(self, name):
    # Get the package directory from build_py
    build_py = self.get_finalized_command('build_py')
    package_dir = build_py.get_package_dir('dynd')
    # This is the name of the dynd-python C-extension
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if (suffix is None):
      suffix = sysconfig.get_config_var('SO')
    filename = name + suffix
    return os.path.join(package_dir, filename)    

  def get_ext_built(self, name):
    if sys.platform != 'win32':
        suffix = sysconfig.get_config_var('EXT_SUFFIX')
        if (suffix is None):
            suffix = sysconfig.get_config_var('SO')
        return name + suffix

    return os.path.join('Release', name + '.pyd')

  def run(self):
    # We don't call the origin build_ext, instead ignore that
    # default behavior and call cmake for DyND's one C-extension.

    # The directory containing this setup.py
    source = dirname(abspath(__file__))

    # The staging directory for the module being built
    build_temp = os.path.join(os.getcwd(), self.build_temp)
    build_lib = os.path.join(os.getcwd(), self.build_lib)

    # Change to the build directory
    saved_cwd = getcwd()
    if not os.path.isdir(self.build_temp):
        self.mkpath(self.build_temp)
    chdir(self.build_temp)

    # Detect if we built elsewhere
    if os.path.isfile('CMakeCache.txt'):
        cachefile = open('CMakeCache.txt', 'r')
        cachedir = re.search('CMAKE_CACHEFILE_DIR:INTERNAL=(.*)', cachefile.read()).group(1)
        cachefile.close()

        if (cachedir != build_temp):
            return

    pyexe_option = '-DPYTHON_EXECUTABLE=%s' % sys.executable
    install_lib_option = '-DDYND_INSTALL_LIB=ON'
    static_lib_option = ''
    build_tests_option = ''
    cuda_option = ''
    # If libdynd is checked out into the libraries subdir,
    # we want to build libdynd as part of dynd-python, not
    # separately like the default does.
    if os.path.isfile(os.path.join(source,
                          'libraries/libdynd/include/dynd/array.hpp')):
        install_lib_option = '-DDYND_INSTALL_LIB=OFF'
        static_lib_option = '-DDYND_SHARED_LIB=OFF'
        build_tests_option = '-DDYND_BUILD_TESTS=OFF'
    else:
        built_with_cuda = eval(check_output(['libdynd-config', '-cuda']))
        if built_with_cuda:
            cuda_option = '-DDYND_CUDA=ON'

    if sys.platform != 'win32':
        self.spawn(['cmake', pyexe_option, install_lib_option,
                    static_lib_option, cuda_option, source])
        self.spawn(['make'])
    else:
        import struct
        # Always build with MSVC 2013 (TODO: 2015 support)
        cmake_generator = 'Visual Studio 12 2013'
        if is_64_bit: cmake_generator += ' Win64'
        # Generate the build files
        self.spawn(['cmake', source, pyexe_option, install_lib_option,
                    static_lib_option, build_tests_option,
                    '-G', cmake_generator])
        # Do the build
        self.spawn(['cmake', '--build', '.', '--config', 'Release'])

    import shutil
    
    # ...
    shutil.move('libpydynd.so', os.path.join(build_lib, 'dynd', 'libpydynd.so'))

    # Move the built C-extension to the place expected by the Python build
    self._found_names = []
    for name in self.get_expected_names():
        built_path = self.get_ext_built(name)
        if os.path.exists(built_path):
            ext_path = os.path.join(build_lib, self.get_ext_path(name))
            if os.path.exists(ext_path):
                os.remove(ext_path)
            self.mkpath(os.path.dirname(ext_path))
            print('Moving built DyND C-extension to build path', ext_path)
            shutil.move(self.get_ext_built(name), ext_path)
            self._found_names.append(name)
        else:
            print('DyND C-extension skipped:', built_path)

    chdir(saved_cwd)

  def get_expected_names(self):
    return ['_pydynd', os.path.join('ndt', 'type'), 'cuda']

  def get_names(self):
    return self._found_names

  def get_outputs(self):
    # Just the C extensions
    return [self.get_ext_path(name) for name in self.get_names()]

if sys.version_info >= (2, 7):
    from subprocess import check_output
else:
    def check_output(args):
        import subprocess
        return subprocess.Popen(args, stdout = subprocess.PIPE).communicate()[0]

# Get the version number to use from git
ver = check_output(['git', 'describe', '--dirty',
                               '--always', '--match', 'v*']).decode('ascii').strip('\n')

# Same processing as in __init__.py
if '.' in ver:
    vlst = ver.lstrip('v').split('.')
    vlst = vlst[:-1] + vlst[-1].split('-')

    if len(vlst) > 3:
        # The 4th one may not be, so trap it
        try:
            # Zero pad the dev version #, so it sorts lexicographically
            vlst[3] = 'dev%03d' % int(vlst[3])
            # increment the third version number, so
            # the '.dev##' versioning convention works
            vlst[2] = str(int(vlst[2]) + 1)
        except ValueError:
            pass
        ver = '.'.join(vlst[:4])
        # Can't use the local version on PyPI, so just exclude this part
        # + '+' + '.'.join(vlst[4:])
    else:
        ver = '.'.join(vlst)

setup(
    name = 'dynd',
    description = 'Python exposure of DyND',
    version = ver,
    author = 'DyND Developers',
    author_email = 'libdynd-dev@googlegroups.com',
    license = 'BSD',
    url = 'https://github.com/libdynd/dynd-python',
    packages = [
        'dynd',
        'dynd.nd',
        'dynd.ndt',
        'dynd._lowlevel',
        'dynd.tests',
    ],
    package_data = {'dynd': ['libpydynd.dylib', '*.pxd', 'include/*.hpp']},
    # build_ext is overridden to call cmake, the Extension is just
    # needed so things like bdist_wheel understand what's going on.
    ext_modules = [Extension('_pydynd', sources=[])],
    # This includes both build and install requirements. Setuptools' setup_requires
    # option does not actually install things, so isn't actually helpful...
    install_requires=open('dev-requirements.txt').read().strip().split('\n'),
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
    ],
    cmdclass = {'build_ext': cmake_build_ext},
)

