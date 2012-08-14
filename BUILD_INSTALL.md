PREREQUISITES
=============

This library requires a C++98 or C++11 compiler. On Windows, Visual
Studio 2010 is the recommended compiler. On Mac OS X, clang
is the recommended compiler. On Linux, gcc 4.6.1 and 4.7.0 have
been tested.

 * dynamicndarray, included as a git submodule

Before configuring the build with CMake, initialize the git submodule
with the following commands from the root blaze directory:

    $ git submodule init
    $ git submodule update

 * Python 2.7
 * Cython >= 0.16
 * Numpy >= 1.5

 * CMake >= 2.6
 * Boost (header-only, doesn't require that any libraries be built)

CONFIGURATION OPTIONS
=====================

These are some options which can be configured by calling
CMake with an argument like "-DCMAKE_BUILD_TYPE=Release".

CMAKE_BUILD_TYPE
    Which kind of build, such as Release, RelWithDebugInfo, Debug.
PYTHON_PACKAGE_INSTALL_PREFIX
    Where the Python module should be installed.
CMAKE_INSTALL_PREFIX
    The prefix for installing shared libraries such as
    libdynamicndarray.so.

BUILD AND INSTALL INSTRUCTIONS
==============================

CMake is the only supported build system for this library. This
may expand in the future, but for the time being this is the
only one which will be kept up to date.

Windows
-------

Visual Studio 2010 or newer is recommended, and works against
Python versions built with previous compilers.

1. Run CMake-gui.

2. For the 'source code' folder, choose the
    dynamicndarray folder which is the root of the project.

3. For the 'build the binaries' folder, create a 'build'
    subdirectory so that your build is isolated from the
    source code files.

4. Click 'Add Entry', and create a variable called BOOST_ROOT.
   Set its type to PATH, and select the path to the boost library.
   Boost doesn't need to be built, only headers are used.

5. Double-click on the generated dynamicndarray.sln
    to open Visual Studio. The RelWithDebInfo configuration is
    recommended for most purposes.

6. To install the Python module, explicitly build the INSTALL target.

*OR*

Start a command prompt window, and navigate to the
dynamicndarray folder which is the root of the project.
Switch the "-G" argument below to "Visual Studio 10" if using
32-bit Python. Replace the "BOOST_ROOT" path with the path to boost
(boost doesn't need to be built, only headers are used).
Execute the following commands:

    D:\dynamicndarray>mkdir build
    D:\dynamicndarray>cd build
    D:\dynamicndarray>set BOOST_ROOT=D:\Develop\boost_1_48_0
    D:\dynamicndarray\build>cmake -G "Visual Studio 10 Win64" ..
       [output, check it for errors]
    D:\dynamicndarray\build>start dynamicndarray.sln
       [Visual Studio should start and load the project]

The RelWithDebInfo configuration is recommended for most purposes.
To install the Python module, explicitly build the INSTALL target.

Linux
-----

Execute the following commands from the dynamicndarray folder,
which is the root of the project:

    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    $ make install # or "sudo make install"

If you want to control where the dynamicndarray shared object is
installed, and where the Python module goes, use the
`CMAKE_INSTALL_PREFIX` and `PYTHON_PACKAGE_INSTALL_PREFIX`
cmake configuration variables respectively.

Mac OS X
--------

Switch the "-DCMAKE\_OSX\_ARCHITECTURES" argument below to "i386" if
you're using 32-bit Python. Execute the following commands
from the dynamicndarray folder, which is the root of the project:

    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_FLAGS="-stdlib=libc++"  ..
    $ make
    $ make install # or "sudo make install"

If you want to control where the dynamicndarray shared object is
installed, and where the Python module goes, use the
`CMAKE_INSTALL_PREFIX` and `PYTHON_PACKAGE_INSTALL_PREFIX`
cmake configuration variables respectively.

