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

REM If no MSVC version is selected, choose 2010
if "%MSVC_VERSION%" == "" set MSVC_VERSION=10.0
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

REM Use conda to create a conda environment of the required
REM python version and containing the dependencies.
SET PYENV_PREFIX=%WORKSPACE%\build\pyenv
C:\Anaconda\python .\buildscripts\create_conda_pyenv_retry.py %PYTHON_VERSION% %PYENV_PREFIX%
IF %ERRORLEVEL% NEQ 0 exit /b 1
echo on
set PYTHON_EXECUTABLE=%PYENV_PREFIX%\Python.exe
set PATH=%PYENV_PREFIX%;%PYENV_PREFIX%\Scripts;%PATH%

REM Select the correct compiler, by first getting whether
REM the Python is 32-bit or 64-bit.
FOR /F "delims=" %%i IN ('%PYTHON_EXECUTABLE% -c "import ctypes;print(8*ctypes.sizeof(ctypes.c_void_p))"') DO set PYTHON_BITS=%%i
if "%PYTHON_BITS%" == "64" goto :python64
 set MSVC_VCVARS_PLATFORM=x86
 set MSVC_BUILD_PLATFORM=Win32
 if "%MSVC_VERSION%" == "9.0" set CMAKE_BUILD_TARGET="Visual Studio 9 2008"
 if "%MSVC_VERSION%" == "10.0" set CMAKE_BUILD_TARGET="Visual Studio 10"
 if "%MSVC_VERSION%" == "11.0" set CMAKE_BUILD_TARGET="Visual Studio 11"
goto :python32
:python64
 set MSVC_VCVARS_PLATFORM=amd64
 set MSVC_BUILD_PLATFORM=x64
 if "%MSVC_VERSION%" == "9.0" set CMAKE_BUILD_TARGET="Visual Studio 9 2008 Win64"
 if "%MSVC_VERSION%" == "10.0" set CMAKE_BUILD_TARGET="Visual Studio 10 Win64"
 if "%MSVC_VERSION%" == "11.0" set CMAKE_BUILD_TARGET="Visual Studio 11 Win64"
:python32

REM Configure the appropriate visual studio command line environment
if "%PROGRAMFILES(X86)%" == "" set VCDIR=%PROGRAMFILES%\Microsoft Visual Studio %MSVC_VERSION%\VC
if NOT "%PROGRAMFILES(X86)%" == "" set VCDIR=%PROGRAMFILES(X86)%\Microsoft Visual Studio %MSVC_VERSION%\VC
call "%VCDIR%\vcvarsall.bat" %MSVC_VCVARS_PLATFORM%
IF %ERRORLEVEL% NEQ 0 exit /b 1
echo on

REM Create a fresh visual studio solution with cmake, and do the build/install
cd build
cmake -DCMAKE_INSTALL_PREFIX=install -G %CMAKE_BUILD_TARGET% -DPYTHON_EXECUTABLE=%PYTHON_EXECUTABLE% ..
IF %ERRORLEVEL% NEQ 0 exit /b 1
devenv dynd-python.sln /Build "RelWithDebInfo|%MSVC_BUILD_PLATFORM%"
IF %ERRORLEVEL% NEQ 0 exit /b 1
devenv dynd-python.sln /Build "RelWithDebInfo|%MSVC_BUILD_PLATFORM%" /Project INSTALL
IF %ERRORLEVEL% NEQ 0 exit /b 1

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

cd ..

exit /b 0