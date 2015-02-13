"""
from distutils.command.build import build
from distutils.command.install import install
from distutils.core import setup
"""

from distutils.command.build import build
from setuptools.command.install import install
from setuptools import setup

from os import chdir
from os.path import abspath, dirname

class cmake_build(build):
  description = "build dynd-python with CMake"

  def run(self):
    self.mkpath(self.build_lib)

    # Store the source directory
    source = dirname(abspath(__file__))

    # Change to the build directory
    chdir(self.build_lib)

    self.spawn(['cmake', source])
    self.spawn(['make'])

class cmake_install(install):
  description = "install dynd-python with CMake"

  def run(self):
    # Change to the build directory
    chdir(self.build_lib)

    self.spawn(['make', 'install'])

setup(name = 'dynd', description = 'Python exposure of dynd', version = '0.6.6',
  cmdclass = {'build': cmake_build, 'install': cmake_install})