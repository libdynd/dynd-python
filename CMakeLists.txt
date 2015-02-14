#
# Copyright (C) 2011-15 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cmake_minimum_required(VERSION 2.8.11)
project(dynd-python)

################################################
# Some options configurable from the CMAKE command execution
#
# -DDYND_INSTALL_LIB=ON/OFF, Use a libdynd which has been built
#   and installed separately. To build with this option off, libdynd
#   must be checked out into the libraries/libdynd subdirectory.
option(DYND_INSTALL_LIB
    "Use a libdynd built and installed somewhere."
    ON)
# -DUSE_RELATIVE_RPATH=ON/OFF, For OSX, to use the @rpath mechanism
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

# Force the default build type to be Release, because a Debug
# build doesn't work properly with the default Python build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif()

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
else()
    # Set some options used by the libdynd CMakeLists.txt
    if(WIN32)
        set(DYND_SHARED_LIB OFF)
    else()
        set(DYND_SHARED_LIB ON)
    endif()
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

    set(LIBDYND_INCLUDE_DIRS
        "libraries/libdynd/include"
        "${CMAKE_CURRENT_BINARY_DIR}/libraries/libdynd/include")
endif()

# Get the git revision
get_git_head_revision("${CMAKE_CURRENT_SOURCE_DIR}" GIT_REFSPEC DYND_PYTHON_GIT_SHA1)
git_describe("${CMAKE_CURRENT_SOURCE_DIR}" DYND_PYTHON_VERSION_STRING --dirty --always --match "v*")
message(STATUS "DyND-Python version: ${DYND_PYTHON_VERSION_STRING}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/git_version.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/src/git_version.cpp" @ONLY)

# Extract the version number from the version string
string(REPLACE "v" "" DYND_PYTHON_VERSION "${DYND_PYTHON_VERSION_STRING}")
string(REPLACE "-" ";" DYND_PYTHON_VERSION "${DYND_PYTHON_VERSION}")
list(GET DYND_PYTHON_VERSION 0 "${DYND_PYTHON_VERSION}")

if(WIN32)
    # Treat warnings as errors (-WX does this)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -WX -EHsc")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 16)
            message(FATAL_ERROR "Only MSVC 2010 (Version 16.0) and later are supported by LibDyND. Found version ${CMAKE_CXX_COMPILER_VERSION}.")
        endif ()
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fomit-frame-pointer -fstrict-aliasing -fPIC -Wall -Wextra -Werror -Wno-missing-field-initializers")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        if ("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 4.7)
            message(FATAL_ERROR "Only GCC 4.7 and later are supported by LibDyND. Found version ${CMAKE_CXX_COMPILER_VERSION}.")
        endif()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmax-errors=20")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ferror-limit=20 -Wdocumentation")
    endif()
endif()

if (CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

include_directories(
    ${NUMPY_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${LIBDYND_INCLUDE_DIRS}
    include/pydynd
    )

set(pydynd_CPP_SRC
    include/pydynd/array_as_numpy.hpp
    include/pydynd/array_as_pep3118.hpp
    include/pydynd/array_as_py.hpp
    include/pydynd/array_assign_from_py.hpp
    include/pydynd/array_from_py.hpp
    include/pydynd/array_from_py_dynamic.hpp
    include/pydynd/array_from_py_typededuction.hpp
    include/pydynd/array_functions.hpp
    include/pydynd/arrfunc_functions.hpp
    include/pydynd/codegen_cache_functions.hpp
    include/pydynd/arrfunc_from_instantiate_pyfunc.hpp
    include/pydynd/arrfunc_from_pyfunc.hpp
    include/pydynd/copy_from_numpy_arrfunc.hpp
    include/pydynd/copy_from_pyobject_arrfunc.hpp
    include/pydynd/copy_to_numpy_arrfunc.hpp
    include/pydynd/copy_to_pyobject_arrfunc.hpp
    include/pydynd/ctypes_interop.hpp
    include/pydynd/do_import_array.hpp
    include/pydynd/elwise_gfunc_functions.hpp
    include/pydynd/elwise_map.hpp
    include/pydynd/elwise_reduce_gfunc_functions.hpp
    include/pydynd/eval_context_functions.hpp
    include/pydynd/exception_translation.hpp
    include/pydynd/gfunc_callable_functions.hpp
    include/pydynd/git_version.hpp
    include/pydynd/init.hpp
    include/pydynd/numpy_interop.hpp
    include/pydynd/numpy_ufunc_kernel.hpp
    include/pydynd/placement_wrappers.hpp
    include/pydynd/py_lowlevel_api.hpp
    include/pydynd/type_functions.hpp
    include/pydynd/utility_functions.hpp
    include/pydynd/vm_elwise_program_functions.hpp
    src/array_as_numpy.cpp
    src/array_as_pep3118.cpp
    src/array_as_py.cpp
    src/array_assign_from_py.cpp
    src/array_functions.cpp
    src/array_from_py.cpp
    src/array_from_py_dynamic.cpp
    src/array_from_py_typededuction.cpp
    src/arrfunc_from_instantiate_pyfunc.cpp
    src/arrfunc_from_pyfunc.cpp
    src/arrfunc_functions.cpp
    src/codegen_cache_functions.cpp
    src/copy_from_numpy_arrfunc.cpp
    src/copy_from_pyobject_arrfunc.cpp
    src/copy_to_numpy_arrfunc.cpp
    src/copy_to_pyobject_arrfunc.cpp
    src/ctypes_interop.cpp
    src/elwise_map.cpp
    src/elwise_gfunc_functions.cpp
    src/elwise_reduce_gfunc_functions.cpp
    src/eval_context_functions.cpp
    src/exception_translation.cpp
    src/gfunc_callable_functions.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/src/git_version.cpp
    src/git_version.cpp.in
    src/init.cpp
    src/numpy_interop.cpp
    src/numpy_ufunc_kernel.cpp
    src/py_lowlevel_api.cpp
    src/type_functions.cpp
    src/utility_functions.cpp
    src/vm_elwise_program_functions.cpp
    )
set(pydynd_CYTHON_SRC
    dynd/dynd._pydynd.pyx
    dynd/array.pxd
    dynd/codegen_cache.pxd
    dynd/dynd.pxd
    dynd/elwise_gfunc.pxd
    dynd/elwise_reduce_gfunc.pxd
    dynd/eval_context.pxd
    dynd/gfunc_callable.pxd
    dynd/ndt_type.pxd
    dynd/vm_elwise_program.pxd
    )
set_source_files_properties(${pydynd_CYTHON_SRC} PROPERTIES CYTHON_IS_CXX 1)

source_group("Cython Source" REGULAR_EXPRESSION ".*pyx$")
source_group("Cython Headers" REGULAR_EXPRESSION ".*pxd$")

cython_add_module(dynd._pydynd ${pydynd_CPP_SRC} ${pydynd_CYTHON_SRC})

set_target_properties(
    dynd._pydynd
    PROPERTIES
    OUTPUT_NAME _pydynd
    LINKER_LANGUAGE CXX)
if(WIN32)
    if(MSVC12)
    set_target_properties(
        dynd._pydynd PROPERTIES
        COMPILE_FLAGS
        "-DHAVE_ROUND"
        )
    endif()
elseif(APPLE)
    # We compile with -Werror, and Cython's code generates warnings, so
    # disable that specifically for this project.
    set_target_properties(
        dynd._pydynd PROPERTIES
        #SOVERSION ${DYND_PYTHON_VERSION}
        #VERSION ${DYND_PYTHON_VERSION}
        COMPILE_FLAGS
        "-Wno-unused-parameter -Wno-unused-function -Wno-error"
        )
    if(USE_RELATIVE_RPATH)
        message(STATUS "Adding relative rpath to _pydynd for linking libdynd")
        # TODO: The rpath used is Anaconda-specific, should calculate
        #       the relative install path instead of using "../../..".
        add_custom_command(TARGET dynd._pydynd
            POST_BUILD COMMAND ${CMAKE_INSTALL_NAME_TOOL}
                -add_rpath "@loader_path/../../.."
                $<TARGET_FILE:dynd._pydynd>)
    else()
        message(STATUS "NOT adding relative rpath to _pydynd for linking libdynd")
    endif()
else()
    # We compile with -Werror, and Cython's code generates warnings, so
    # disable that specifically for this project.
    set_target_properties(
        dynd._pydynd
        PROPERTIES
        COMPILE_FLAGS
        "-Wno-error"
        )
    if (USE_RELATIVE_RPATH)
        set_target_properties(
            dynd._pydynd
            PROPERTIES
            # TODO: The rpath used is Anaconda-specific, should make sure it
            #       works more generally.
            INSTALL_RPATH
            "$ORIGIN/../../.."
            )
    endif()
endif()


if (DYND_INSTALL_LIB)
    target_link_libraries(dynd._pydynd "${LIBDYND_LIBRARIES}")
else()
    target_link_libraries(dynd._pydynd libdynd)
endif()

# Install all the Python scripts
install(DIRECTORY dynd DESTINATION "${PYTHON_PACKAGE_INSTALL_PREFIX}"
    FILES_MATCHING PATTERN "*.py")
# Install the module
install(TARGETS dynd._pydynd DESTINATION "${PYTHON_PACKAGE_INSTALL_PREFIX}/dynd")
install(DIRECTORY "include" DESTINATION "${PYTHON_PACKAGE_INSTALL_PREFIX}/dynd")
# Install the libdynd binary if on unix and libdynd wasn't built separately
if(NOT DYND_INSTALL_LIB AND NOT WIN32)
    get_property(dynd_loc TARGET libdynd PROPERTY LOCATION)
    install(FILES ${dynd_loc} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
