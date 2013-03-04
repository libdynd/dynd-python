#
# Copyright (C) 2011-13, DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.6)
project(dynd-python)

# For the Git SHA1/version code
list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/libraries/dynd/cmake/")
include(GetGitRevisionDescriptionDyND)

# Force the default build type to be Release, because a Debug
# build doesn't work properly with the default Python build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(PythonInterp REQUIRED)
find_package(PythonLibsNew REQUIRED)
find_package(NumPy REQUIRED)
include(UseCython)

# Default install location for Python packages
if (NOT PYTHON_PACKAGE_INSTALL_PREFIX)
    set(PYTHON_PACKAGE_INSTALL_PREFIX "${PYTHON_SITE_PACKAGES}" CACHE STRING
      "Choose the Python module directory (default site-packages)" FORCE)
endif()

# Require version >= 1.5
if(NUMPY_VERSION_DECIMAL LESS 10500)
    message(FATAL_ERROR,
        "Blaze dynd requires NumPy >= 1.5")
endif()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/libraries/dynd/include/dynd/ndobject.hpp")
message(FATAL_ERROR
    "The dynd C++ library must be placed in\n"
    "libraries/dynd.\nRemove any temporary CMake\n"
    "files, then if you're using git, run\n"
    "'git clone git@github.com:ContinuumIO/dynd.git'\n"
    "from the libraries directory.\n"
    "See BUILD_INSTALL.txt for more details.")
endif()

# Include dynd in the build
add_subdirectory(libraries/dynd)

# Get the git revision
get_git_head_revision("${CMAKE_CURRENT_SOURCE_DIR}" GIT_REFSPEC DYND_PYTHON_GIT_SHA1)
git_describe("${CMAKE_CURRENT_SOURCE_DIR}" DYND_PYTHON_VERSION_STRING --dirty)
message(STATUS "DyND-Python version: ${DYND_PYTHON_VERSION_STRING}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/git_version.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/src/git_version.cpp" @ONLY)

if(WIN32)
    # Treat warnings as errors (-WX does this)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -WX -EHsc")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fomit-frame-pointer -fstrict-aliasing -fPIC -Wall -Wextra -Werror")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        # The '-fmax-errors' flag was first supported in gcc 4.6
        # Use the c++0x flag only in 4.6 and later
        if (NOT "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS "4.6")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -fmax-errors=20")
        endif()
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ferror-limit=20")
        # Apple has done this weird thing where its Clang reports a different
        # version than the official Clang.
        # TODO: someone more OS X-savvy will have to fine-tune this.
        if (APPLE OR "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS "3.2")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wdocumentation")
        endif()
    endif()
endif()

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIRS}
    libraries/dynd/include
    include
    )

set(pydynd_CPP_SRC
    include/codegen_cache_functions.hpp
    include/ctypes_interop.hpp
    include/do_import_array.hpp
    include/placement_wrappers.hpp
    include/dtype_functions.hpp
    include/elwise_map.hpp
    include/exception_translation.hpp
    include/gfunc_callable_functions.hpp
    include/git_version.hpp
    include/ndobject_functions.hpp
    include/ndobject_from_py.hpp
    include/ndobject_as_pep3118.hpp
    include/ndobject_as_numpy.hpp
    include/ndobject_as_py.hpp
    include/numpy_interop.hpp
    include/elwise_gfunc_functions.hpp
    include/elwise_reduce_gfunc_functions.hpp
    include/utility_functions.hpp
    include/vm_elwise_program_functions.hpp
    src/codegen_cache_functions.cpp
    src/ctypes_interop.cpp
    src/dtype_functions.cpp
    src/elwise_map.cpp
    src/gfunc_callable_functions.cpp
    src/exception_translation.cpp
    src/ndobject_functions.cpp
    src/ndobject_from_py.cpp
    src/ndobject_as_pep3118.cpp
    src/ndobject_as_numpy.cpp
    src/ndobject_as_py.cpp
    src/numpy_interop.cpp
    src/elwise_gfunc_functions.cpp
    src/elwise_reduce_gfunc_functions.cpp
    src/git_version.cpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/src/git_version.cpp
    src/utility_functions.cpp
    src/vm_elwise_program_functions.cpp
    )
set(pydynd_CYTHON_SRC
    src/_pydynd.pyx
    include/codegen_cache.pxd
    include/dynd.pxd
    include/dtype.pxd
    include/ndobject.pxd
    include/gfunc_callable.pxd
    include/elwise_gfunc.pxd
    include/elwise_reduce_gfunc.pxd
    include/vm_elwise_program.pxd
    )
set_source_files_properties(${pydynd_CYTHON_SRC} PROPERTIES CYTHON_IS_CXX 1)

source_group("Cython Source" REGULAR_EXPRESSION ".*pyx$")
source_group("Cython Headers" REGULAR_EXPRESSION ".*pxd$")

cython_add_module(_pydynd ${pydynd_CPP_SRC} ${pydynd_CYTHON_SRC})

set_target_properties(_pydynd PROPERTIES LINKER_LANGUAGE CXX)
if(WIN32)
elseif(APPLE)
    # We compile with -Werror, and Cython's code generates warnings, so
    # disable that specifically for this project.
    set_target_properties(
        _pydynd
        PROPERTIES COMPILE_FLAGS
        "-Wno-unused-parameter -Wno-self-assign -Wno-unused-function -Wno-error"
        )
else()
    # strict aliasing is disabled because Cython generated code
    # which made g++ 4.6.1 complain
    set_target_properties(
        _pydynd
        PROPERTIES COMPILE_FLAGS
        "-Wno-unused-parameter -Wno-unused-function -fno-strict-aliasing"
        )
endif()

target_link_libraries(_pydynd dynd)

# Install all the Python scripts
install(DIRECTORY dynd DESTINATION "${PYTHON_PACKAGE_INSTALL_PREFIX}"
    FILES_MATCHING PATTERN "*.py")
# Install the module
install(TARGETS _pydynd DESTINATION "${PYTHON_PACKAGE_INSTALL_PREFIX}/dynd")
# Install the basic_kernels .dll/.so
get_property(basic_kernels_loc TARGET basic_kernels PROPERTY LOCATION)
# NOTE: This only uses one configuration for the whole Visual Studio
#       solution, so CMAKE_BUILD_TYPE must match the build target
#       you want to install with.
string(REPLACE "$(Configuration)" ${CMAKE_BUILD_TYPE} basic_kernels_loc ${basic_kernels_loc})
string(REPLACE "$(OutDir)" ${CMAKE_BUILD_TYPE} basic_kernels_loc ${basic_kernels_loc})
install(FILES ${basic_kernels_loc} DESTINATION "${PYTHON_PACKAGE_INSTALL_PREFIX}/dynd/nd")
# Install the dynd lib
if(NOT WIN32)
    get_property(dynd_loc TARGET dynd PROPERTY LOCATION)
    install(FILES ${dynd_loc} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
