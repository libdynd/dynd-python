STEP BY STEP BUILD AND INSTALL
==============================

1. Check the C++ compiler version.

Ensure you have a suitable C++98 or C++11 compiler. On Windows, Visual
Studio 2010 is the recommended compiler, but 2008 has been tested
as well. On Mac OS X, clang is the recommended compiler. On Linux,
gcc 4.6.1, gcc 4.7.0, and clang 3.3-svn have been tested.

2. Get the prerequisites.
  * CMake >= 2.6
  * Python 2.6 or 2.7
  * Cython >= 0.16
  * NumPy >= 1.5
  * git (for cloning the github repositories)

3. Get the source code.

  Check out the dynd-python and dynd source code. The following commands
should work equivalently on Windows and Unix-like operating systems.

  ```
  ~ $ git clone https://github.com/ContinuumIO/dynd-python
  Cloning into dynd-python...
  ~ $ cd dynd-python
  ~/dynd-python $ mkdir libraries
  ~/dynd-python $ cd libraries
  ~/dynd-python/libraries $ git clone https://github.com/ContinuumIO/dynd
  Cloning into dynd...
  ~/dynd-python/libraries $ cd ..
  ~/dynd-python $ mkdir build
  ```

4. Use cmake to create the build files.

  **(Windows)** Run CMake-gui. For the 'source code' folder, choose the
dynd-python folder which is the root of the project. For the
'build the binaries' folder, choose the 'build' subdirectory
created during step 3.

  If you want to control where the installation goes, you can edit
the `CMAKE_INSTALL_PREFIX` and `PYTHON_PACKAGE_INSTALL_PREFIX`
variables in the GUI after clicking 'Configure', then clicking
'Configure' again to update.

  Click on 'Configure' and then 'Generate' to create
dynd-python.sln in the 'build' subdirectory.

  **(OS X)** Run cmake as follows. This describes the 64-bit build,
for a 32-bit build switch the "-DCMAKE\_OSX\_ARCHITECTURES"
argument below to "i386".

  If you want to control where the dynd shared object is
installed, and where the Python module goes, add
`-DCMAKE_INSTALL_PREFIX=<prefix>` and
`-DPYTHON_PACKAGE_INSTALL_PREFIX=<site-pkg-dir>`
to the cmake command.

  ```
  ~/dynd-python $ cd build
  ~/dynd-python/build $ cmake -DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_FLAGS="-stdlib=libc++" -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
  ```

  **(Linux)** Run cmake as follows.

  If you want to control where the dynd shared object is
installed, and where the Python module goes, add
`-DCMAKE_INSTALL_PREFIX=<prefix>` and
`-DPYTHON_PACKAGE_INSTALL_PREFIX=<site-pkg-dir>`
to the cmake command.

  ```
  ~/dynd-python $ cd build
  ~/dynd-python/build $ cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
  ```

5. Run the build and install.

  **(Windows)** Double-click on the 'dynd-python\build\dynd-python.sln'
file to open up Visual Studio. Select 'Release' or 'RelWithDebInfo'
if you're building for release, and build. To install the targets,
right click on the INSTALL project and build it.

  **(OS X and Linux)** From the build directory, run the following.

  ```
  ~/dynd-python/build $ make
  ~/dynd-python/build $ make install # or "sudo make install"
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

