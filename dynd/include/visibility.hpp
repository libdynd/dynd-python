#pragma once
// Symbol visibility macros
#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(_MSC_VER)
// Disable warnings about automatically exporting symbols from the C++
// standard library. We want to get rid of these cases in the long run,
// but for now it's a necessary evil.
#pragma warning( disable : 4251 )
#else
#pragma GCC diagnostic ignored "-Wattributes"
#endif
#if defined(PYDYND_EXPORT) // Building the library
#define PYDYND_API __declspec(dllexport)
#else // Importing the library
#define PYDYND_API __declspec(dllimport)
#define PYDYND_USING_DLL
#endif // defined(PYDYND_EXPORT)
#define PYDYND_INTERNAL
#else
#define PYDYND_API
#define PYDYND_INTERNAL __attribute__ ((visibility ("hidden")))
#endif // End symbol visibility macros.
