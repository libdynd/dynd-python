STEP BY STEP BUILD AND INSTALL
==============================

1. Check the C++ compiler version.

  Ensure you have a suitable C++98 or C++11 compiler. On Windows, Visual
Studio 2010 is the minimum supported compiler. On Mac OS X, clang is the
recommended compiler. On Linux, gcc 4.6.1, gcc 4.7.0, and
clang 3.3-svn have been tested.

2. Get the prerequisites.
  * CMake >= 2.8.11
  * Python 2.6, 2.7, 3.3, or 3.4
  * Cython >= 0.21
  * NumPy >= 1.5
  * git (for cloning the github repositories)
  * Nose (Only for generating xunit .xml output when running tests)

3. Get the source code.

  Check out the dynd-python and libdynd source code. The following commands
should work equivalently on Windows and Unix-like operating systems.

  ```
  ~ $ git clone https://github.com/libdynd/libdynd
  Cloning into libdynd...
  ~ $ git clone https://github.com/libdynd/dynd-python
  Cloning into dynd-python...
  ```

4. Build and install libdynd

  ```
  ~ $ cd libdynd
  ~/libdynd $ mkdir build
  ~/libdynd $ cd build
  ~/libdynd/build $ cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
  <...>
  ~/libdynd/build $ make
  <...>
  ~/libdynd/build $ sudo make install
  ```

  If you want to control where the libdynd shared object is
installed, add `-DCMAKE_INSTALL_PREFIX=<prefix>`
to the cmake command.

4. Build and install dynd-python

  ```
  ~ $ cd dynd-python
  ~/dynd-python $ mkdir build
  ~/dynd-python $ cd build
  ~/dynd-python/build $ cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
  <...>
  ~/dynd-python/build $ make
  <...>
  ~/dynd-python/build $ cd ..
  ~/dynd-python $ python setup.py install
  ```

  If you want to control where the Python module goes, add
`-DPYTHON_PACKAGE_INSTALL_PREFIX=<site-pkg-dir>`
to the cmake command.

COMBINED BUILDS
===============

In some development environments, such as MSVC or XCode, you may prefer
to develop libdynd and dynd-python together in one project. This is
supported by including libdynd in the "libraries/libdynd" subdirectory
of dynd-python, and running cmake with "-DDYND_INSTALL_LIB=OFF".
For example, to set this up on Windows with MSVC 2013, do:

  ```
  D:\>git clone https://github.com/libdynd/dynd-python
  Cloning into 'dynd-python'...
  <...>

  D:\>cd dynd-python

  D:\dynd-python>mkdir libraries

  D:\dynd-python>cd libraries

  D:\dynd-python\libraries>git clone https://github.com/libdynd/libdynd
  Cloning into 'libdynd'...
  <...>

  D:\dynd-python\libraries>cd ..

  D:\dynd-python>mkdir build

  D:\dynd-python>cd build

  D:\dynd-python\build>cmake -DDYND_INSTALL_LIB=OFF -G"Visual Studio 12 Win64" ..
  -- The C compiler identification is MSVC 18.0.21005.1
  <...>

  D:\dynd-python\build>start dynd-python.sln
  <launches MSVC>

  ```

ALTERNATIVE COMPILERS
=====================

  If you want to build with a different compiler, for
  example to use the static analyzer in clang, you can
  customize the compiler at this step.

  ```
  ~/libdynd $ mkdir build-analyze
  ~/libdynd $ cd build-analyze
  ~/libdynd/build-analyze $ export CCC_CC=clang
  ~/libdynd/build-analyze $ export CCC_CXX=clang++
  ~/libdynd/build-analyze $ cmake -DCMAKE_CXX_COMPILER=c++-analyzer -DCMAKE_C_COMPILER=ccc-analyzer ..
  ```

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

