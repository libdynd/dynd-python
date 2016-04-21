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

BUILD FOR DEVELOPMENT
---------------------

Development of DyND should be done using the build configuration combining
libdynd and the DyND Python bindings in a single combined build in a
development mode. With this configuration, a single build will update both
libdynd and dynd-python in a way that both the C++ and Python tests can
be run with no further installation steps.

This combined build configuration works both with make-style builds and
MSVC solution files. The following instructions are the same for both
Windows and Linux/OS X, but on a unix platform run `make` instead of
loading the `dynd-python.sln` file at the end.

  ```
  C:\>git clone --recursive https://github.com/libdynd/dynd-python
  Cloning into 'dynd-python'...
  <...>
  C:\>cd dynd-python
  C:\dynd-python>git clone --recursive https://github.com/libdynd/libdynd
  Cloning into 'libdynd'...
  <...>
  C:\dynd-python>python setup.py develop
  <...>
  C:\dynd-python>cd build-dev
  C:\dynd-python\build-dev>start dynd-python.sln
  ```

BUILD FOR INSTALLATION
----------------

3. Get the source code.

  Check out the dynd-python and libdynd source code. The following commands
should work equivalently on Windows and Unix-like operating systems.

  ```
  ~ $ git clone --recursive https://github.com/libdynd/libdynd
  Cloning into libdynd...
  ~ $ git clone --recursive https://github.com/libdynd/dynd-python
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

