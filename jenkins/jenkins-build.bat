REM
REM Copyright (C) 2011-13, DyND Developers
REM BSD 2-Clause License, see LICENSE.txt
REM
REM This is the master windows build + test script for building
REM the dynd python bindings on jenkins.
REM
REM Jenkins Requirements:
REM   - Anaconda should be installed and in the path
REM   - Use a jenkins build matrix for multiple
REM     platforms/python versions
REM   - Use the XShell plugin to launch this script
REM   - Call the script from the root workspace
REM     directory as ./jenkins/jenkins-build
REM

REM Jenkins has '/' in its workspace. Fix it to '\' to simplify the DOS commands.
set WORKSPACE=%WORKSPACE:/=\%

if "%PROCESSOR_ARCHITECTURE%" == "AMD64" goto :amd64
 set MSVC_VERSION=10.0
 set MSVC_VCVARS_PLATFORM=x86
 set MSVC_BUILD_PLATFORM=Win32
 set CMAKE_BUILD_TARGET="Visual Studio 10"
goto :notamd64
:amd64
 set MSVC_VERSION=10.0
 set MSVC_VCVARS_PLATFORM=amd64
 set MSVC_BUILD_PLATFORM=x64
 set CMAKE_BUILD_TARGET="Visual Studio 10 Win64"
:notamd64

REM Configure the appropriate visual studio command line environment
if "%PROGRAMFILES(X86)%" == "" set VCDIR=%PROGRAMFILES%\Microsoft Visual Studio %MSVC_VERSION%\VC
if NOT "%PROGRAMFILES(X86)%" == "" set VCDIR=%PROGRAMFILES(X86)%\Microsoft Visual Studio %MSVC_VERSION%\VC
call "%VCDIR%\vcvarsall.bat" %MSVC_VCVARS_PLATFORM%
IF %ERRORLEVEL% NEQ 0 exit /b 1
echo on

REM Remove the build subdirectory from last time
rd /q /s build

REM Get libdynd into the libraries subdirectory
REM TODO: Build libdynd in a separate jenkins project,
REM       and use its build artifact here.
call .\jenkins\checkout_libdynd.bat
IF %ERRORLEVEL% NEQ 0 exit /b 1

REM Use conda to create a conda environment of the required
REM python version and containing the dependencies.
SET PYENV_PREFIX=%WORKSPACE%\build\pyenv
call .\jenkins\create_conda_pyenv.bat 2.7 %PYENV_PREFIX%
IF %ERRORLEVEL% NEQ 0 exit /b 1
echo on
export PATH=%PYENV_PREFIX%;%PYENV_PREFIX%\Scripts;%PATH%

REM Create a fresh visual studio solution with cmake, and do the build/install
cd build
cmake -DCMAKE_INSTALL_PREFIX=install -G %CMAKE_BUILD_TARGET% ..
IF %ERRORLEVEL% NEQ 0 exit /b 1
devenv dynd-python.sln /Build "RelWithDebInfo|%MSVC_BUILD_PLATFORM%"
IF %ERRORLEVEL% NEQ 0 exit /b 1
devenv dynd-python.sln /Build "RelWithDebInfo|%MSVC_BUILD_PLATFORM%" /Project INSTALL
IF %ERRORLEVEL% NEQ 0 exit /b 1

REM Run the tests and generate xml results
python -c "import dynd;dynd.test(xunitfile='../test_results.xml', exit=1)"
IF %ERRORLEVEL% NEQ 0 exit /b 1

REM Get the version number and process it into a suitable form
FOR /F "delims=" %%i IN ('python -c "import dynd;print(dynd.__version_string__)"') DO set PYDYND_VERSION=%%i
set PYDYND_VERSION=%PYDYND_VERSION:-=_%
set PYDYND_VERSION=%PYDYND_VERSION:~1%

RM Put the conda package by itself in the directory pkgs
cd ..
rd /q /s pkgs
mkdir pkgs
cd pkgs

# Create a conda package from the build
call conda package -p %PYENV_PREFIX% --pkg-name=dynd --pkg-version=%PYDYND_VERSION%
IF %ERRORLEVEL% NEQ 0 exit /b 1
echo on

cd ..

exit /b 0