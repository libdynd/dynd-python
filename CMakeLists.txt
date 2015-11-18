#
# Copyright (C) 2011-15 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.8.11)
project(dynd-python)

set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_OSX_DEPLOYMENT_TARGET 10.9)
set(CMAKE_VERBOSE_MAKEFILE 1)

################################################
# Some options configurable from the CMAKE command execution
#
# -DDYND_INSTALL_LIB=ON/OFF, Use a libdynd which has been built
#   and installed separately. To build with this option off, libdynd
#   must be checked out into the libraries/libdynd subdirectory.
option(DYND_INSTALL_LIB
    "Use a libdynd built and installed somewhere."
    ON)
# -DUSE_RELATIVE_RPATH=ON/OFF, For Linux and OSX, to use the @rpath mechanism
#   for creating a build which is linked with relative paths. The
#   libdynd should have been built with -DUSE_RELATIVE_RPATH=ON as well.
if(UNIX)
    option(USE_RELATIVE_RPATH
        "Linux/OSX: Add a relative rpath for libdynd to the dynd python extension module."
        OFF)
endif()
################################################

# For the Git SHA1/version code
list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
include(GetGitRevisionDescriptionDyND)

find_package(CUDA 6.5)
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
if (NUMPY_VERSION_DECIMAL LESS 10500)
    message(FATAL_ERROR,
        "DyND-Python requires NumPy >= 1.5")
endif()

if (DYND_INSTALL_LIB)
    find_package(LibDyND REQUIRED)
    if(LIBDYND_CUDA)
        option(DYND_CUDA
            "Use a libdynd built with CUDA. If DYND_INSTALL_LIB is ON, this defaults to the DYND_CUDA option used when building libdynd. Otherwise it defaults to OFF."
            ON)
    else()
        option(DYND_CUDA
            "Use a libdynd built with CUDA. If DYND_INSTALL_LIB is ON, this defaults to the DYND_CUDA option used when building libdynd. Otherwise it defaults to OFF."
            OFF)
    endif()
else()
    set(DYND_SHARED_LIB ON)
    # USE_RELATIVE_RPATH is inherited from this cmakelists, so need to set it here
    option(DYND_BUILD_TESTS "Build the googletest unit tests for libdynd." ON)
    option(DYND_CUDA
        "Use a libdynd built with CUDA. If DYND_INSTALL_LIB is ON, this defaults to the DYND_CUDA option used building that library. Otherwise it defaults to OFF."
        OFF)

    if (NOT EXISTS "${PROJECT_SOURCE_DIR}/libraries/libdynd/include/dynd/array.hpp")
        message(FATAL_ERROR
            "The libdynd C++ library must be placed in libraries/libdynd."
            "Remove any temporary CMake"
            "files, then if you're using git, run"
            "'git clone git@github.com:libdynd/libdynd.git'"
            "from the libraries directory."
            "See BUILD_INSTALL.md for more details.")
    endif()

    # Include libdynd in the build
    add_subdirectory(libraries/libdynd)

    set(LIBDYND_INCLUDE_DIRS
        "libraries/libdynd/include"
        "${CMAKE_CURRENT_BINARY_DIR}/libraries/libdynd/include")
endif()

# Get the git revision
get_git_head_revision("${CMAKE_CURRENT_SOURCE_DIR}" GIT_REFSPEC DYND_PYTHON_GIT_SHA1)
git_describe("${CMAKE_CURRENT_SOURCE_DIR}" DYND_PYTHON_VERSION_STRING --dirty --always --match "v*")
message(STATUS "DyND-Python version: ${DYND_PYTHON_VERSION_STRING}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/dynd/src/git_version.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/dynd/src/git_version.cpp" @ONLY)

# Extract the version number from the version string
string(REPLACE "v" "" DYND_PYTHON_VERSION "${DYND_PYTHON_VERSION_STRING}")
string(REPLACE "-" ";" DYND_PYTHON_VERSION "${DYND_PYTHON_VERSION}")
list(GET DYND_PYTHON_VERSION 0 "${DYND_PYTHON_VERSION}")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # Treat warnings as errors (-WX does this)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -WX -EHsc")
    if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 18)
        message(FATAL_ERROR "Only MSVC 2013 (Version 18.0) and later are supported by LibDyND. Found version ${CMAKE_CXX_COMPILER_VERSION}.")
    endif ()
else()
    if(WIN32)
        # Don't use the -fPIC flag since it is the default on MinGW.
        # Doing so results in a warning that is then raised as an error.
        # Define _hypot=hypot to avoid the conflict between the macro
        # used in the Python headers and the name used in the standard library.
        if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
            # Define MS_WIN64 so that npy_intp has the correct size and
            # the proper module import functions are called on 64 bit Windows.
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_hypot=hypot -std=c++14 -DMS_WIN64 -g -fomit-frame-pointer -fstrict-aliasing -Wall -Wextra -Werror -Wno-missing-field-initializers")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_hypot=hypot -std=c++14 -g -fomit-frame-pointer -fstrict-aliasing -Wall -Wextra -Werror -Wno-missing-field-initializers")
        endif()
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fomit-frame-pointer -fstrict-aliasing -fPIC -Wall -Wextra -Werror -Wno-missing-field-initializers")
    endif()
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        if ("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 4.9)
            message(FATAL_ERROR "Only GCC 4.9 and later are supported by DyND. Found version ${CMAKE_CXX_COMPILER_VERSION}.")
        endif()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fmax-errors=20")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -ferror-limit=20 -Wdocumentation")
    endif()
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_ROUND")
else()
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    if(APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-unused-function -Wno-error")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error")
    endif()
endif()

include_directories(
    ${NUMPY_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${LIBDYND_INCLUDE_DIRS}
    dynd/include
    ${CMAKE_CURRENT_BINARY_DIR}/dynd
    dynd
    )

if(DYND_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DDYND_CUDA)
    set(DYND_CUDA_01 1)
else()
    set(DYND_CUDA_01 0)
endif()

set(pydynd_SRC
    dynd/include/array_as_numpy.hpp
    dynd/include/array_as_pep3118.hpp
    dynd/include/array_as_py.hpp
    dynd/include/array_assign_from_py.hpp
    dynd/include/array_from_py.hpp
    dynd/include/array_from_py_dynamic.hpp
    dynd/include/array_from_py_typededuction.hpp
    dynd/include/array_functions.hpp
    dynd/include/arrfunc_functions.hpp
    dynd/include/arrfunc_from_pyfunc.hpp
    dynd/include/config.hpp
    dynd/include/copy_from_numpy_arrfunc.hpp
    dynd/include/copy_from_pyobject_arrfunc.hpp
    dynd/include/copy_to_numpy_arrfunc.hpp
    dynd/include/copy_to_pyobject_arrfunc.hpp
    dynd/include/ctypes_interop.hpp
    dynd/include/do_import_array.hpp
    dynd/include/eval_context_functions.hpp
    dynd/include/exception_translation.hpp
    dynd/include/gfunc_callable_functions.hpp
    dynd/include/git_version.hpp
    dynd/include/init.hpp
    dynd/include/numpy_interop.hpp
    dynd/include/numpy_ufunc_kernel.hpp
    dynd/include/type_functions.hpp
    dynd/include/utility_functions.hpp
    dynd/include/visibility.hpp
    dynd/include/wrapper.hpp
    dynd/src/array_as_numpy.cpp
    dynd/src/array_as_pep3118.cpp
    dynd/src/array_as_py.cpp
    dynd/src/array_assign_from_py.cpp
    dynd/src/array_functions.cpp
    dynd/src/array_from_py.cpp
    dynd/src/array_from_py_dynamic.cpp
    dynd/src/array_from_py_typededuction.cpp
    dynd/src/arrfunc_from_pyfunc.cpp
    dynd/src/arrfunc_functions.cpp
    dynd/src/copy_from_numpy_arrfunc.cpp
    dynd/src/copy_from_pyobject_arrfunc.cpp
    dynd/src/copy_to_numpy_arrfunc.cpp
    dynd/src/copy_to_pyobject_arrfunc.cpp
    dynd/src/ctypes_interop.cpp
    dynd/src/eval_context_functions.cpp
    dynd/src/exception_translation.cpp
    dynd/src/gfunc_callable_functions.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/dynd/src/git_version.cpp
    dynd/src/git_version.cpp.in
    dynd/src/init.cpp
    dynd/src/numpy_interop.cpp
    dynd/src/numpy_ufunc_kernel.cpp
    dynd/src/type_functions.cpp
    dynd/src/utility_functions.cpp
    dynd/src/wrapper.cpp
)

add_library(pydynd SHARED ${pydynd_SRC})
set_property(
   TARGET pydynd
   PROPERTY COMPILE_DEFINITIONS PYDYND_EXPORT
   )

if(APPLE)
    set_target_properties(pydynd PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
elseif(WIN32)
    target_link_libraries(pydynd "${PYTHON_LIBRARIES}")
endif()
if(DYND_INSTALL_LIB)
    target_link_libraries(pydynd "${LIBDYND_LIBRARIES}")
else()
    target_link_libraries(pydynd libdynd)
endif()

foreach(module config eval_context ndt.type ndt.json nd.array nd.callable nd.functional nd.registry)
    string(REPLACE "." ";" directories ${module})
    list(GET directories -1 module_name)
    list(REMOVE_AT directories -1)

    string(REPLACE "." "/" module_root "${module}")
    set(module_SRC dynd/${module_root}.pyx)
    set_source_files_properties(${module_SRC} PROPERTIES CYTHON_IS_CXX 1)

    cython_add_module(${module_name} ${module_SRC})

    if (directories)
        string(REPLACE ";" "/" module_output_directory ${directories})
        set_target_properties(${module_name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${module_output_directory})
    endif()

    if(APPLE)
        set(module_install_rpath "@loader_path")
    else()
        set(module_install_rpath "$ORIGIN")
    endif()
    list(LENGTH directories i)
    while(${i} GREATER 0)
        set(module_install_rpath "${module_install_rpath}/..")
        math(EXPR i "${i} - 1" )
    endwhile(${i} GREATER 0)
#    set(module_install_rpath "/opt/anaconda/envs/_test/lib/python2.7/site-packages/dynd/")
    set_target_properties(${module_name} PROPERTIES INSTALL_RPATH ${module_install_rpath})

    target_link_libraries(${module_name} pydynd)
endforeach(module)
