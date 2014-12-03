REM
REM Copyright (C) 2011-13, DyND Developers
REM BSD 2-Clause License, see LICENSE.txt
REM
REM This is the master windows build + test script for building
REM the dynd python bindings on jenkins.
REM
REM Jenkins Requirements:
REM   - Anaconda should be installed in C:\Anaconda.
REM   - Use a jenkins build matrix for multiple
REM     platforms/python versions
REM   - Use the XShell plugin to launch this script
REM   - Call the script from the root workspace
REM     directory as buildscripts/jenkins-build
REM   - Use a user-defined axis to select python versions with PYTHON_VERSION
REM

REM If no MSVC version is selected, choose 2013
if "%MSVC_VERSION%" == "" set MSVC_VERSION=12.0
REM Require a version of Python to be selected
if "%PYTHON_VERSION%" == "" exit /b 1

REM Jenkins has '/' in its workspace. Fix it to '\' to simplify the DOS commands.
set WORKSPACE=%WORKSPACE:/=\%

REM Remove the build subdirectory from last time
rd /q /s build

REM Get libdynd into the libraries subdirectory
REM TODO: Build libdynd in a separate jenkins project,
REM       and use its build artifact here.
call .\buildscripts\checkout_libdynd.bat
IF %ERRORLEVEL% NEQ 0 exit /b 1

REM Make sure binstar is installed in the main environment
echo Updating binstar...
call C:\Anaconda\Scripts\conda install --yes binstar || exit 1
call C:\Anaconda\Scripts\binstar --version
echo on

REM Use conda to create a conda environment of the required
REM python version and containing the dependencies.
SET PYENV_PREFIX=%WORKSPACE%\build\pyenv
rd /s /q %PYENV_PREFIX%
REM NOTE: cython is forced to 0.20.2 temporarily because of a bug in anaconda
REM       https://github.com/libdynd/anaconda-issues/issues/178
call C:\Anaconda\Scripts\conda create --yes -p %PYENV_PREFIX% python=%PYTHON_VERSION% cython=0.20.2 scipy nose
IF %ERRORLEVEL% NEQ 0 exit /b 1
echo on
set PYTHON_EXECUTABLE=%PYENV_PREFIX%\Python.exe
set PATH=%PYENV_PREFIX%;%PYENV_PREFIX%\Scripts;%PATH%

REM Select the correct compiler, by first getting whether
REM the Python is 32-bit or 64-bit.
FOR /F "delims=" %%i IN ('%PYTHON_EXECUTABLE% -c "import ctypes;print(8*ctypes.sizeof(ctypes.c_void_p))"') DO set PYTHON_BITS=%%i
if "%PYTHON_BITS%" == "64" goto :python64
 if "%MSVC_VERSION%" == "12.0" set CMAKE_BUILD_TARGET="Visual Studio 12"
goto :python32
:python64
 if "%MSVC_VERSION%" == "12.0" set CMAKE_BUILD_TARGET="Visual Studio 12 Win64"
:python32

REM Create a fresh visual studio solution with cmake, and do the build/install
cd build
cmake -DDYND_INSTALL_LIB=OFF -DCMAKE_INSTALL_PREFIX=install -G %CMAKE_BUILD_TARGET% -DPYTHON_EXECUTABLE=%PYTHON_EXECUTABLE% .. || exit /b 1
cmake --build . --config RelWithDebInfo || exit /b 1
cmake --build . --config RelWithDebInfo --target install || exit /b 1

REM Run the tests and generate xml results
%PYTHON_EXECUTABLE% -c "import dynd;dynd.test(xunitfile='../test_results.xml', verbosity=2, exit=1)"
IF %ERRORLEVEL% NEQ 0 exit /b 1

REM Get the version number and process it into a suitable form
FOR /F "delims=" %%i IN ('%PYTHON_EXECUTABLE% -c "import dynd;print(dynd.__version__)"') DO set PYDYND_VERSION=%%i
if "%PYDYND_VERSION%" == "" exit /b 1
set PYDYND_VERSION=%PYDYND_VERSION:-=_%

REM Put the conda package by itself in the directory pkgs/<anaconda-arch>
cd ..
rd /q /s pkgs
mkdir pkgs
cd pkgs
mkdir win-%PYTHON_BITS%
cd win-%PYTHON_BITS%

REM Create a conda package from the build
call C:\Anaconda\Scripts\conda package -p %PYENV_PREFIX% --pkg-name=dynd-python --pkg-version=%PYDYND_VERSION%
IF %ERRORLEVEL% NEQ 0 exit /b 1
echo on

REM Upload the package to binstar
FOR /F "delims=" %%i IN ('dir /b dynd-python-*.tar.bz2') DO set PKG_FILE=%%i
call C:\Anaconda\Scripts\binstar -t %BINSTAR_MWIEBE_AUTH% upload --force %PKG_FILE% || exit 1
call C:\Anaconda\Scripts\binstar -t %BINSTAR_BLAZE_AUTH% upload --force %PKG_FILE% || exit 1

cd ..

exit /b 0
