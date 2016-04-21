STEP BY STEP BUILD AND INSTALL
==============================

1. Check the C++ compiler version.

  Ensure you have a suitable C++14 compiler. On Windows, Visual
Studio 2015 is the minimum supported compiler. On Mac OS X, clang is the
recommended compiler. On Linux, gcc 4.9 and later, and
clang 3.4 and later have been tested.

2. Get the prerequisites.
  * CMake >= 2.8.11
  * Python 2.7, 3.4, or 3.5
  * Cython >= 0.24
  * NumPy >= 1.7.1
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
  ~/dynd-python $ python setup.py install
  ```

  If you want to control where the Python module goes, add
`-DPYTHON_PACKAGE_INSTALL_PREFIX=<site-pkg-dir>`
to the cmake command.

COMBINED BUILDS
===============

In some development environments, such as MSVC or XCode, you may prefer
to develop libdynd and dynd-python together in one project. This is
supported by including libdynd in the "libdynd" subdirectory
of dynd-python, and running cmake with "-DDYND_INSTALL_LIB=OFF".
For example, to set this up on Windows with MSVC 2015, do:

  ```
  D:\>git clone https://github.com/libdynd/dynd-python
  Cloning into 'dynd-python'...
  <...>

  D:\>cd dynd-python

  D:\dynd-python>git clone https://github.com/libdynd/libdynd
  Cloning into 'libdynd'...
  <...>

  D:\dynd-python>python setup.py install

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
CMAKE_INSTALL_PREFIX
    The prefix for installing shared libraries such as
    libdynd.so.

