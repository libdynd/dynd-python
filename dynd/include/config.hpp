#pragma once

#include <Python.h>

#include <dynd/config.hpp>
#include <dynd/array.hpp>

#if defined(_WIN32)
#if defined(PYDYND_EXPORT)
// Building the library
#define PYDYND_API __declspec(dllexport)
#else
// Importing the library
#define PYDYND_API __declspec(dllimport)
#endif
#else
#define PYDYND_API
#endif