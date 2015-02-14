"""
from distutils.command.build import build
from distutils.command.install import install
from distutils.core import setup
"""

from distutils.command.build_ext import build_ext
from distutils import sysconfig
from setuptools.command.install import install
from setuptools import setup, Extension

import os, sys
from os import chdir, getcwd
from os.path import abspath, dirname

# Check if we're running 64-bit Python
import struct
is_64_bit = struct.calcsize('@P') == 8

class cmake_build_ext(build_ext):
  description = "Build the C-extension for dynd-python with CMake"

  def get_dyndext_path(self):
    # Get the package directory from build_py
    build_py = self.get_finalized_command('build_py')
    package_dir = build_py.get_package_dir('dynd')
    # This is the name of the dynd-python C-extension
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if (suffix is None):
      suffix = sysconfig.get_config_var('SO')
    filename = '_pydynd' + suffix
    return os.path.join(package_dir, filename)

  def run(self):
    # We don't call the origin build_ext, instead ignore that
    # default behavior and call cmake for DyND's one C-extension.

    # The directory containing this setup.py
    source = dirname(abspath(__file__))

    # The staging directory for the module being built
    build_lib = os.path.join(os.getcwd(), self.build_lib)

    # Change to the build directory
    saved_cwd = getcwd()
    self.mkpath(self.build_temp)
    chdir(self.build_temp)

    pyexe_option = '-DPYTHON_EXECUTABLE=%s' % sys.executable
    install_lib_option = '-DDYND_INSTALL_LIB=ON'
    static_lib_option = ''
    # If libdynd is checked out into the libraries subdir,
    # we want to build libdynd as part of dynd-python, not
    # separately like the default does.
    if os.path.isfile(os.path.join(source,
                          'libraries/libdynd/include/dynd/array.hpp')):
        install_lib_option = '-DDYND_INSTALL_LIB=OFF'
        static_lib_option = '-DDYND_SHARED_LIB=OFF'

    if sys.platform != 'win32':
        self.spawn(['cmake', pyexe_option, install_lib_option,
                    static_lib_option, source])
        self.spawn(['make'])
        suffix = sysconfig.get_config_var('EXT_SUFFIX')
        if (suffix is None):
          suffix = sysconfig.get_config_var('SO')
        dyndext_built = '_pydynd' + suffix
    else:
        import struct
        # Always build with MSVC 2013 (TODO: 2015 support)
        cmake_generator = 'Visual Studio 12 2013'
        if is_64_bit: cmake_generator += ' Win64'
        # Generate the build files
        self.spawn(['cmake', source, pyexe_option, install_lib_option,
                    static_lib_option, '-G', cmake_generator])
        # Do the build
        self.spawn(['cmake', '--build', '.', '--config', 'Release'])
        dyndext_built = 'Release/_pydynd.pyd'
    # Move the built C-extension to the place expected by the Python build
    import shutil
    dyndext_path = os.path.join(build_lib, self.get_dyndext_path())
    if os.path.exists(dyndext_path):
        os.remove(dyndext_path)
    self.mkpath(os.path.dirname(dyndext_path))
    print('Moving built DyND C-extension to build path', dyndext_path)
    shutil.move(dyndext_built, dyndext_path)
    chdir(saved_cwd)

  def get_outputs(self):
    # Just the C-extension
    return [self.get_dyndext_path()]


setup(
    name = 'dynd',
    description = 'Python exposure of DyND',
    version = '0.6.6',
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
    # build_ext is overridden to call cmake, the Extension is just
    # needed so things like bdist_wheel understand what's going on.
    ext_modules = [Extension('dynd._pydynd', sources=[])],
    install_requires=open('requirements.txt').read().strip().split('\n'),
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
    ],
    cmdclass = {'build_ext': cmake_build_ext},
)

