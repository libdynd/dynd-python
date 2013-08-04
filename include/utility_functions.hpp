//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__UTILITY_FUNCTIONS_HPP_
#define _DYND__UTILITY_FUNCTIONS_HPP_

#include "Python.h"

#include <sstream>
#include <stdexcept>
#include <string>

#include <dynd/type.hpp>

namespace pydynd {

/**
 * Function which casts the parameter to
 * a PyObject pointer and calls Py_XDECREF on it.
 */
void py_decref_function(void* obj);

/**
 * A container class for managing the local lifetime of
 * PyObject *.
 *
 * Throws an exception if the object passed into the constructor
 * is NULL.
 */
class pyobject_ownref {
    PyObject *m_obj;

    // Non-copyable
    pyobject_ownref(const pyobject_ownref&);
    pyobject_ownref& operator=(const pyobject_ownref&);
public:
    inline pyobject_ownref()
        : m_obj(NULL)
    {
    }
    inline explicit pyobject_ownref(PyObject* obj)
        : m_obj(obj)
    {
        if (obj == NULL) {
            throw std::runtime_error("propagating a Python exception...");
        }
    }

    inline pyobject_ownref(PyObject* obj, bool inc_ref)
        : m_obj(obj)
    {
        if (obj == NULL) {
            throw std::runtime_error("propagating a Python exception...");
        }
        if (inc_ref) {
            Py_INCREF(m_obj);
        }
    }

    inline ~pyobject_ownref()
    {
        Py_XDECREF(m_obj);
    }

    /**
     * Resets the reference owned by this object to the one provided.
     * This steals a reference to the input parameter, 'obj'.
     *
     * \param obj  The reference to replace the current one in this object.
     */
    inline void reset(PyObject* obj)
    {
        if (obj == NULL) {
            throw std::runtime_error("propagating a Python exception...");
        }
        Py_XDECREF(m_obj);
        m_obj = obj;
    }

    /**
     * Clears the owned reference to NULL.
     */
    inline void clear()
    {
        Py_XDECREF(m_obj);
        m_obj = NULL;
    }

    /** Returns a borrowed reference. */
    inline PyObject *get() const
    {
        return m_obj;
    }

    /** Returns a borrowed reference. */
    inline operator PyObject *()
    {
        return m_obj;
    }

    /**
     * Returns the reference owned by this object,
     * use it like "return obj.release()". After the
     * call, this object contains NULL.
     */
    inline PyObject *release()
    {
        PyObject *result = m_obj;
        m_obj = NULL;
        return result;
    }
};

class PyGILState_RAII {
    PyGILState_STATE m_gstate;

    PyGILState_RAII(const PyGILState_RAII&);
    PyGILState_RAII& operator=(const PyGILState_RAII&);
public:
    inline PyGILState_RAII() {
        m_gstate = PyGILState_Ensure();
    }

    inline ~PyGILState_RAII() {
        PyGILState_Release(m_gstate);
    }
};

size_t pyobject_as_size_t(PyObject *obj);
intptr_t pyobject_as_index(PyObject *index);
int pyobject_as_int_index(PyObject *index);
dynd::irange pyobject_as_irange(PyObject *index);
std::string pystring_as_string(PyObject *str);

void pyobject_as_vector_ndt_type(PyObject *list_dtype, std::vector<dynd::ndt::type>& vector_dtype);
void pyobject_as_vector_string(PyObject *list_string, std::vector<std::string>& vector_string);
void pyobject_as_vector_intp(PyObject *list_index, std::vector<intptr_t>& vector_intp,
                bool allow_int);
void pyobject_as_vector_int(PyObject *list_int, std::vector<int>& vector_int);

/**
 * Same as PySequence_Size, but throws a C++
 * exception on error.
 */
inline Py_ssize_t pysequence_size(PyObject *seq)
{
    Py_ssize_t s = PySequence_Size(seq);
    if (s == -1 && PyErr_Occurred()) {
        throw std::exception();
    }
    return s;
}

/**
 * Same as PyDict_GetItemString, but throws a
 * C++ exception on error.
 */
inline PyObject *pydict_getitemstring(PyObject *dp, const char *key)
{
    PyObject *result = PyDict_GetItemString(dp, key);
    if (result == NULL) {
        throw std::exception();
    }
    return result;
}

PyObject* intptr_array_as_tuple(size_t size, const intptr_t *array);

/**
 * Parses the axis argument, which may be either a single index
 * or a tuple of indices. They are converted into a boolean array
 * which is set to true whereever a reduction axis is provided.
 *
 * Returns the number of axes which were set.
 */
int pyarg_axis_argument(PyObject *axis, int ndim, dynd::dynd_bool *reduce_axes);

/**
 * Parses the error_mode argument. If it is None, returns
 * assign_error_default.
 */
dynd::assign_error_mode pyarg_error_mode(PyObject *error_mode_obj);

/**
 * Matches the input object against one of several
 * strings, returning the corresponding integer.
 */
int pyarg_strings_to_int(PyObject *obj, const char *argname, int default_value,
                const char *string0, int value0);

/**
 * Matches the input object against one of several
 * strings, returning the corresponding integer.
 */
int pyarg_strings_to_int(PyObject *obj, const char *argname, int default_value,
                const char *string0, int value0,
                const char *string1, int value1);

int pyarg_strings_to_int(PyObject *obj, const char *argname, int default_value,
                const char *string0, int value0,
                const char *string1, int value1,
                const char *string2, int value2);

int pyarg_strings_to_int(PyObject *obj, const char *argname, int default_value,
                const char *string0, int value0,
                const char *string1, int value1,
                const char *string2, int value2,
                const char *string3, int value3);

int pyarg_strings_to_int(PyObject *obj, const char *argname, int default_value,
                const char *string0, int value0,
                const char *string1, int value1,
                const char *string2, int value2,
                const char *string3, int value3,
                const char *string4, int value4);

bool pyarg_bool(PyObject *obj, const char *argname, bool default_value);


uint32_t pyarg_access_flags(PyObject* obj);

} // namespace pydynd

#endif // _DYND__UTILITY_FUNCTIONS_HPP_
