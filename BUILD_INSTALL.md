PREREQUISITES
=============

This library requires a C++98 or C++11 compiler. On Windows, Visual
Studio 2010 is the recommended compiler, but 2008 has been tested
as well. On Mac OS X, clang is the recommended compiler. On Linux,
gcc 4.6.1 and 4.7.0 have been tested.

 * https://github.com/ContinuumIO/dynd

Before configuring the build with CMake, clone the dynd project
into the libraries subdirectory of dynd-python with the following
commands:

    (dynd-python)$ mkdir libraries
    (dynd-python)$ cd libraries
    (dynd-python/libraries) $ git clone https://github.com/ContinuumIO/dynd

 * Python 2.7
 * Cython >= 0.16
 * Numpy >= 1.5

 * CMake >= 2.6

CONFIGURATION OPTIONS
=====================

These are some options which can be configured by calling
CMake with an argument like "-DCMAKE_BUILD_TYPE=Release".

CMAKE_BUILD_TYPE
    Which kind of build, such as Release, RelWithDebInfo, Debug.
PYTHON_PACKAGE_INSTALL_PREFIX
    Where the Python module should be installed.
CMAKE_INSTALL_PREFIX
    The prefix for installing shared libraries such as
    libdynd.so.

BUILD AND INSTALL INSTRUCTIONS
==============================

CMake is the only supported build system for this library. This
may expand in the future, but for the time being this is the
only one which will be kept up to date.

Windows
-------

Visual Studio 2010 or newer is recommended, and works against
Python versions built with previous compilers. You can use
either the CMake gui program or its command line tools.

1. Run CMake-gui.

2. For the 'source code' folder, choose the
    dynd-python folder which is the root of the project.

3. For the 'build the binaries' folder, create a 'build'
    subdirectory so that your build is isolated from the
    source code files.

4. Double-click on the generated dynd-python.sln
    to open Visual Studio. The RelWithDebInfo configuration is
    recommended for most purposes.

5. To install the Python module, explicitly build the INSTALL target.

*OR*

Start a command prompt window, and navigate to the
dynd-python folder which is the root of the project.
Switch the "-G" argument below to "Visual Studio 10" if using
32-bit Python.
Execute the following commands:

    D:\dynd-python>mkdir build
    D:\dynd-python>cd build
    D:\dynd-python\build>cmake -G "Visual Studio 10 Win64" -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
       [output, check it for errors]
    D:\dynd-python\build>start dynd-python.sln
       [Visual Studio should start and load the project]

The RelWithDebInfo configuration is recommended for most purposes.
To install the Python module, explicitly build the INSTALL target.

Linux
-----

Execute the following commands from the dynd-python folder,
which is the root of the project (Replace RelWithDebInfo with
Release if doing a release build that doesn't need debug info):

    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
    $ make
    $ make install # or "sudo make install"

If you want to control where the dynd shared object is
installed, and where the Python module goes, use the
`CMAKE_INSTALL_PREFIX` and `PYTHON_PACKAGE_INSTALL_PREFIX`
cmake configuration variables respectively.

You may have to customize some library locations, for example a
build configuration on a customized centos 5 install might
look like this:

    $ cmake -DCMAKE_C_COMPILER=/usr/bin/gcc44 -DCMAKE_CXX_COMPILER=/usr/bin/g++44 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPythonInterp_FIND_VERSION=2.6 ..

Mac OS X
--------

Switch the "-DCMAKE\_OSX\_ARCHITECTURES" argument below to "i386" if
you're using 32-bit Python. Execute the following commands
from the dynd-python folder, which is the root of the project
(Replace RelWithDebInfo with Release if doing a release build
that doesn't need debug info):

    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_FLAGS="-stdlib=libc++" -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
    $ make
    $ make install # or "sudo make install"

If you want to control where the dynd shared object is
installed, and where the Python module goes, use the
`CMAKE_INSTALL_PREFIX` and `PYTHON_PACKAGE_INSTALL_PREFIX`
cmake configuration variables respectively.

