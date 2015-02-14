"""
from distutils.command.build import build
from distutils.command.install import install
from distutils.core import setup
"""

from distutils.command.build import build
from setuptools.command.install import install
from setuptools import setup

import os, sys
from os import chdir
from os.path import abspath, dirname

# Check if we're running 64-bit Python
import struct
is_64_bit = len(struct.pack('P', 0)) == 8

class cmake_build(build):
  description = "build dynd-python with CMake"

  def run(self):
    self.mkpath(self.build_lib)

    # Store the source directory
    source = dirname(abspath(__file__))

    # Change to the build directory
    chdir(self.build_lib)

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
        self.spawn(['cmake', source, pyexe_option, install_lib_option,
                    static_lib_option])
        self.spawn(['make'])
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

class cmake_install(install):
  description = "install dynd-python with CMake"

  def run(self):
    # Change to the build directory
    chdir(self.build_lib)

    if sys.platform != 'win32':
        self.spawn(['make', 'install'])
    else:
        self.spawn(['cmake', '--build', '.', '--config', 'Release',
                    '--target', 'install'])

setup(name = 'dynd', description = 'Python exposure of dynd', version = '0.6.6',
  cmdclass = {'build': cmake_build, 'install': cmake_install})
