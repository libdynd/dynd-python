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
# -DDYND_INSTALL_LIB=ON/OFF, Use a libdynd which has been built and
#   installed separately. To build with this option off, libdynd
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

# When this is enabled, the cmake build assumes it is in the directory
# <dynd-python>/build-dev and copies all binaries to the necessary
# place for in-place development of the Python bindings to work.
option(DYND_PYTHON_INPLACE_BUILD
       "Enabled via `python setup.py develop`"
       OFF)

# For the Git SHA1/version code
list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
include(GetGitRevisionDescriptionDyND)

find_package(PythonInterp REQUIRED)
find_package(PythonLibsNew REQUIRED)
find_package(NumPy REQUIRED)
include(UseCython)
include(PostprocessCython)

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

# Helper function for controlling the output directory in in-place build
# mode. By setting the per-target properties, this works on all platforms.
function(set_target_library_output_dir target dir)
    set_target_properties(${target} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${dir}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${dir})
    set_target_properties(${target} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${dir}
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${dir})
    set_target_properties(${target} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${dir}
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${dir})
    set_target_properties(${target} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${dir}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${dir})
endfunction()

if (DYND_INSTALL_LIB)
    find_package(LibDyND REQUIRED)
else()
    set(DYND_SHARED_LIB ON)
    # USE_RELATIVE_RPATH is inherited from this cmakelists, so need to set it here
    option(DYND_BUILD_TESTS "Build the googletest unit tests for libdynd." ON)

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

    set(LIBDYND_INCLUDE_DIR
        "libraries/libdynd/include"
        "${CMAKE_CURRENT_BINARY_DIR}/libraries/libdynd/include")

    if (DYND_PYTHON_INPLACE_BUILD)
        # Build the libdynd.so/.dll directly to the in-place Python bindings
        set_target_library_output_dir(libdynd "${PROJECT_SOURCE_DIR}/dynd")
        set_target_library_output_dir(libdyndt "${PROJECT_SOURCE_DIR}/dynd")
    endif()
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

if(MSVC)
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

if(MSVC)
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
    ${LIBDYND_INCLUDE_DIR}
    dynd/include
    ${CMAKE_CURRENT_BINARY_DIR}/dynd/nd
    ${CMAKE_CURRENT_BINARY_DIR}/dynd/ndt
    ${CMAKE_CURRENT_BINARY_DIR}
    )

foreach(pyx_api_file dynd/nd/array.pyx dynd/nd/callable.pyx dynd/ndt/type.pyx)
    set_source_files_properties(${pyx_api_file} PROPERTIES CYTHON_API 1)
    set_source_files_properties(${pyx_api_file} PROPERTIES CYTHON_PUBLIC 1)
endforeach(pyx_api_file)

set_source_files_properties(dynd/config.pyx PROPERTIES CYTHON_API 1)

cython_add_module(dynd.nd.array dynd.nd.array_pyx True
                  # Additional C++ source files:
                  dynd/include/do_import_array.hpp
                  dynd/include/numpy_interop.hpp
                  dynd/src/array_as_pep3118.cpp
                  dynd/src/array_as_numpy.cpp
                  dynd/src/array_from_py.cpp
                  dynd/src/assign.cpp
                  dynd/src/array_conversions.cpp
                  dynd/src/copy_from_numpy_arrfunc.cpp
                  dynd/src/init.cpp
                  dynd/src/functional.cpp
                  dynd/src/numpy_interop.cpp
                  dynd/src/type_conversions.cpp
                  dynd/src/type_deduction.cpp
                  dynd/src/types/pyobject_type.cpp
                  )
if (DYND_PYTHON_INPLACE_BUILD)
    set_target_library_output_dir(dynd.nd.array "${PROJECT_SOURCE_DIR}/dynd/nd")
endif()

cython_add_module(dynd.config dynd.config_pyx True
                  dynd/include/exception_translation.hpp
                  dynd/src/array_conversions.cpp
                  dynd/src/type_conversions.cpp
                  dynd/src/type_deduction.cpp
                  ${CMAKE_CURRENT_BINARY_DIR}/dynd/src/git_version.cpp
                  )
if (DYND_PYTHON_INPLACE_BUILD)
    set_target_library_output_dir(dynd.config "${PROJECT_SOURCE_DIR}/dynd")
endif()

foreach(module dynd.ndt.type dynd.ndt.json dynd.nd.callable dynd.nd.functional dynd.nd.registry)
    cython_add_module(${module} ${module}_pyx True
                      # Additional C++ source files:
                      dynd/src/type_conversions.cpp
                      dynd/src/array_conversions.cpp)
endforeach(module)
if (DYND_PYTHON_INPLACE_BUILD)
    set_target_library_output_dir(dynd.ndt.type "${PROJECT_SOURCE_DIR}/dynd/ndt")
    set_target_library_output_dir(dynd.ndt.json "${PROJECT_SOURCE_DIR}/dynd/ndt")
    set_target_library_output_dir(dynd.nd.callable "${PROJECT_SOURCE_DIR}/dynd/nd")
    set_target_library_output_dir(dynd.nd.functional "${PROJECT_SOURCE_DIR}/dynd/nd")
    set_target_library_output_dir(dynd.nd.registry "${PROJECT_SOURCE_DIR}/dynd/nd")
endif()


# Run a postprocess script to work around some Cython bugs
# that haven't been fixed in the latest release.
postprocess_cython( postprocess.py dynd.ndt.type_postprocess dynd.ndt.type_pyx dynd.ndt.type)
postprocess_cython( postprocess.py dynd.nd.array_postprocess dynd.nd.array_pyx dynd.nd.array)
postprocess_cython( postprocess.py dynd.nd.callable_postprocess dynd.nd.callable_pyx dynd.nd.callable)

foreach(module dynd.config dynd.ndt.type dynd.ndt.json dynd.nd.array dynd.nd.callable dynd.nd.functional dynd.nd.registry)
    # Temporarily continue to define PYDYND_EXPORT to avoid inconsistent linkage warnings.
    # This should be removed once the macros have been refactored to hide all symbols
    # other than module initialization routines.
    set_property(
        TARGET ${module}
        PROPERTY COMPILE_DEFINITIONS PYDYND_EXPORT
    )
    # Make sure the api headers are all built and postprocessed
    # before anything tries to build conversions.cpp
    add_dependencies(${module} dynd.nd.array_postprocess dynd.nd.callable_postprocess dynd.ndt.type_postprocess)
    if(DYND_INSTALL_LIB)
        target_link_libraries(${module} "${LIBDYND_LIBRARIES}")
    else()
        target_link_libraries(${module} libdynd libdyndt)
        if(UNIX)
            # Make sure libdynd is on the rpath.
            # On Windows, the dll is loaded dynamically via the logic in
            # dynd/__init__.py
            if(APPLE)
                set(module_install_rpath "@loader_path")
            else()
                set(module_install_rpath "$ORIGIN")
            endif()
            string(REPLACE "." ";" module_directories "${module}")
            list(LENGTH module_directories i)
            while(${i} GREATER 2)
                set(module_install_rpath "${module_install_rpath}/..")
                math(EXPR i "${i} - 1" )
            endwhile(${i} GREATER 2)
            set_target_properties(${module} PROPERTIES INSTALL_RPATH ${module_install_rpath})
        endif()
    endif()
endforeach(module)
