# - Find the LibDyND library
# This module finds the LibDyND library which was built separately
# and installed.
#
#  LIBDYND_FOUND               - was LibDyND found
#  LIBDYND_LIBRARIES           - full paths for the LibDyND libraries
#  LIBDYND_LIBRARY_DIR         - the directory containing the libdynd libraries
#  LIBDYND_LIBRARY_NAMES       - the names of the libdynd libraries
#  LIBDYND_INCLUDE_DIR         - path to the LibDyND include files
#  LIBDYND_ROOT_DIR            - directory containing LIBDYND_LIBRARY_DIR and LIBDYND_INCLUDE_DIR
#  LIBDYND_VERSION             - the version of LibDyND found as a string

#============================================================================
# Copyright 2013 Continuum Analytics, Inc.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
#============================================================================

# Try to find a libdynd-config program
if(WIN32)
    # Aside from the system path, search some other locations.
    find_program(_LIBDYND_CONFIG "libdynd-config.bat"
        PATHS
            "C:\\Program Files\\LibDyND\\bin"
            "C:\\Program Files (x86)\\LibDyND\\bin"
            "D:\\Program Files\\LibDyND\\bin"
            "D:\\Program Files (x86)\\LibDyND\\bin"
        )
else()
    find_program(_LIBDYND_CONFIG "libdynd-config")
endif()

if("${_LIBDYND_CONFIG}" STREQUAL "")

    if(LibDyND_FIND_REQUIRED)
        message(FATAL_ERROR "Failed to find libdynd-config program")
    endif()
    set(LIBDYND_FOUND False)
    return()

else()

    # Get the dyndt library to link against.
    execute_process(COMMAND "${_LIBDYND_CONFIG}" "-libdyndtname"
                    RESULT_VARIABLE _DYND_SEARCH_SUCCESS
                    OUTPUT_VARIABLE LIBDYNDT_NAME
                    ERROR_VARIABLE _DYND_ERROR_VALUE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _DYND_SEARCH_SUCCESS MATCHES 0)
        message(FATAL_ERROR "Error getting dyndt library name:\n${_DYND_ERROR_VALUE}")
    endif()

    # Get the libraries to link against.
    execute_process(COMMAND "${_LIBDYND_CONFIG}" "-libnames"
                    RESULT_VARIABLE _DYND_SEARCH_SUCCESS
                    OUTPUT_VARIABLE LIBDYND_LIBRARY_NAMES
                    ERROR_VARIABLE _DYND_ERROR_VALUE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _DYND_SEARCH_SUCCESS MATCHES 0)
        message(FATAL_ERROR "Error getting dynd library names:\n${_DYND_ERROR_VALUE}")
    endif()

    execute_process(COMMAND "${_LIBDYND_CONFIG}" "-libdir"
                    RESULT_VARIABLE _DYND_SEARCH_SUCCESS
                    OUTPUT_VARIABLE LIBDYND_LIBRARY_DIR
                    ERROR_VARIABLE _DYND_ERROR_VALUE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _DYND_SEARCH_SUCCESS MATCHES 0)
        message(FATAL_ERROR "Error getting dynd library directory:\n${_DYND_ERROR_VALUE}")
    endif()

    # Construct DYND_LIBRARIES from the names and directory given.
    if(WIN32)
        string(REPLACE "\\" "/" LIBDYND_LIBRARY_DIR ${LIBDYND_LIBRARY_DIR})
    endif()

    set(LIBDYNDT_LIBRARY "${LIBDYND_LIBRARY_DIR}/${LIBDYNDT_NAME}")

    set(LIBDYND_LIBRARIES "")
    foreach(_lib ${LIBDYND_LIBRARY_NAMES})
        LIST(APPEND LIBDYND_LIBRARIES "${LIBDYND_LIBRARY_DIR}/${_lib}")
    endforeach()

    # Get the include directory
    execute_process(COMMAND "${_LIBDYND_CONFIG}" "-includedir"
        RESULT_VARIABLE _DYND_SEARCH_SUCCESS
        OUTPUT_VARIABLE LIBDYND_INCLUDE_DIR
        ERROR_VARIABLE _DYND_ERROR_VALUE
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _DYND_SEARCH_SUCCESS MATCHES 0)
        message(FATAL_ERROR "Error getting dynd include directory:\n${_DYND_ERROR_VALUE}")
    endif()

    # Get the root directory
    execute_process(COMMAND "${_LIBDYND_CONFIG}" "-rootdir"
        RESULT_VARIABLE _DYND_SEARCH_SUCCESS
        OUTPUT_VARIABLE LIBDYND_ROOT_DIR
        ERROR_VARIABLE _DYND_ERROR_VALUE
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _DYND_SEARCH_SUCCESS MATCHES 0)
        message(FATAL_ERROR "Error getting dynd root directory:\n${_DYND_ERROR_VALUE}")
    endif()

    # Get the version
    execute_process(COMMAND "${_LIBDYND_CONFIG}" "-version"
                    RESULT_VARIABLE _DYND_SEARCH_SUCCESS
                    OUTPUT_VARIABLE LIBDYND_VERSION
                    ERROR_VARIABLE _DYND_ERROR_VALUE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _DYND_SEARCH_SUCCESS MATCHES 0)
        message(FATAL_ERROR "Error getting dynd version:\n${_DYND_ERROR_VALUE}")
    endif()

    find_package_message(LIBDYND "Found LibDyND: version \"${LIBDYND_VERSION}\"" "${LIBDYND_ROOT_DIR}.")

    set(LIBDYND_FOUND True)
endif()
