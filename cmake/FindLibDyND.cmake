# - Find the LibDyND library
# This module finds the LibDyND library which was built separately
# and installed.
#
#  LIBDYND_FOUND               - was LibDyND found
#  LIBDYND_VERSION             - the version of LibDyND found as a string
#  LIBDYND_LIBRARIES           - path to the LibDyND library
#  LIBDYND_INCLUDE_DIRS        - path to the LibDyND include files
#  LIBDYND_CUDA                - if LibDyND was built with cuda support

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
        message(FATAL_ERROR
            "Failed to find libdynd-config program")
    endif()
    set(LIBDYND_FOUND FALSE)
    return()
endif()


# Get the version
execute_process(COMMAND "${_LIBDYND_CONFIG}" "-version"
    RESULT_VARIABLE _DYND_SEARCH_SUCCESS
    OUTPUT_VARIABLE LIBDYND_VERSION
    ERROR_VARIABLE _DYND_ERROR_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _DYND_SEARCH_SUCCESS MATCHES 0)
    message(FATAL_ERROR
        "Error getting additional properties of libdynd:\n${_DYND_ERROR_VALUE}")
endif()

# Get the library to link against
execute_process(COMMAND "${_LIBDYND_CONFIG}" "-libpath"
    RESULT_VARIABLE _DYND_SEARCH_SUCCESS
    OUTPUT_VARIABLE LIBDYND_LIBRARIES
    ERROR_VARIABLE _DYND_ERROR_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _DYND_SEARCH_SUCCESS MATCHES 0)
    message(FATAL_ERROR
        "Error getting additional properties of libdynd:\n${_DYND_ERROR_VALUE}")
endif()

# Get the include directory
execute_process(COMMAND "${_LIBDYND_CONFIG}" "-includedir"
    RESULT_VARIABLE _DYND_SEARCH_SUCCESS
    OUTPUT_VARIABLE LIBDYND_INCLUDE_DIRS
    ERROR_VARIABLE _DYND_ERROR_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _DYND_SEARCH_SUCCESS MATCHES 0)
    message(FATAL_ERROR
        "Error getting additional properties of libdynd:\n${_DYND_ERROR_VALUE}")
endif()

# Get whether or not libdynd was built with cuda support
execute_process(COMMAND "${_LIBDYND_CONFIG}" "-cuda"
    RESULT_VARIABLE _DYND_SEARCH_SUCCESS
    OUTPUT_VARIABLE LIBDYND_CUDA
    ERROR_VARIABLE _DYND_ERROR_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _DYND_SEARCH_SUCCESS MATCHES 0)
    message(FATAL_ERROR
        "Error getting additional properties of libdynd:\n${_DYND_ERROR_VALUE}")
endif()
# Verify that the value in LIBDYND_CUDA is either "ON" or "OFF"
if(NOT ("${LIBDYND_CUDA}" STREQUAL "ON" OR "${LIBDYND_CUDA}" STREQUAL "OFF"))
    message(FATAL_ERROR "Unrecognized cuda option returned from libdynd-config.")
endif()

find_package_message(LIBDYND
    "Found LibDyND: version \"${LIBDYND_VERSION}\",  ${LIBDYND_LIBRARIES}"
    "${LIBDYND_INCLUDE_DIRS}${LIBDYND_LIBRARIES}${LIBDYND_VERSION}")

set(NUMPY_FOUND TRUE)

