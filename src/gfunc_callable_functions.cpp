//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "gfunc_callable_functions.hpp"
#include "utility_functions.hpp"
#include "ndobject_functions.hpp"
#include "ndobject_as_py.hpp"
#include "placement_wrappers.hpp"

#include <dynd/dtypes/fixedstruct_dtype.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

PyTypeObject *pydynd::WNDObjectCallable_Type;

void pydynd::init_w_ndobject_callable_typeobject(PyObject *type)
{
    WNDObjectCallable_Type = (PyTypeObject *)type;
}


void pydynd::add_dtype_names_to_dir_dict(const dtype& dt, PyObject *dict)
{
    if (dt.extended()) {
        const std::pair<std::string, gfunc::callable> *properties;
        int count;
        dt.extended()->get_dynamic_dtype_properties(&properties, &count);
        for (int i = 0; i < count; ++i) {
            if (PyDict_SetItemString(dict, properties[i].first.c_str(), Py_None) < 0) {
                throw runtime_error("");
            }
        }
    }
}

PyObject *pydynd::get_dtype_dynamic_property(const dynd::dtype& dt, PyObject *name)
{
    if (dt.extended()) {
        const std::pair<std::string, gfunc::callable> *properties;
        int count;
        dt.extended()->get_dynamic_dtype_properties(&properties, &count);
        // TODO: We probably want to make some kind of acceleration structure for the name lookup
        if (count > 0) {
            string nstr = pystring_as_string(name);
            for (int i = 0; i < count; ++i) {
                if (properties[i].first == nstr) {
                    return call_gfunc_callable(nstr, properties[i].second, dt);
                }
            }
        }
    }

    PyErr_SetObject(PyExc_AttributeError, name);
    return NULL;
}

void pydynd::add_ndobject_names_to_dir_dict(const dynd::ndobject& n, PyObject *dict)
{
    dtype dt = n.get_dtype();
    if (dt.extended()) {
        const std::pair<std::string, gfunc::callable> *properties;
        int count;
        // Add the ndobject properties
        dt.extended()->get_dynamic_ndobject_properties(&properties, &count);
        for (int i = 0; i < count; ++i) {
            if (PyDict_SetItemString(dict, properties[i].first.c_str(), Py_None) < 0) {
                throw runtime_error("");
            }
        }
        // Add the ndobject functions
        dt.extended()->get_dynamic_ndobject_functions(&properties, &count);
        for (int i = 0; i < count; ++i) {
            if (PyDict_SetItemString(dict, properties[i].first.c_str(), Py_None) < 0) {
                throw runtime_error("");
            }
        }
    }
}

PyObject *pydynd::get_ndobject_dynamic_property(const dynd::ndobject& n, PyObject *name)
{
    dtype dt = n.get_dtype();
    if (dt.extended()) {
        const std::pair<std::string, gfunc::callable> *properties;
        int count;
        // Search for a property
        dt.extended()->get_dynamic_ndobject_properties(&properties, &count);
        // TODO: We probably want to make some kind of acceleration structure for the name lookup
        if (count > 0) {
            string nstr = pystring_as_string(name);
            for (int i = 0; i < count; ++i) {
                if (properties[i].first == nstr) {
                    return call_gfunc_callable(nstr, properties[i].second, n);
                }
            }
        }
        // Search for a function
        dt.extended()->get_dynamic_ndobject_functions(&properties, &count);
        if (count > 0) {
            string nstr = pystring_as_string(name);
            for (int i = 0; i < count; ++i) {
                if (properties[i].first == nstr) {
                    return wrap_ndobject_callable(nstr, properties[i].second, n);
                }
            }
        }
    }

    PyErr_SetObject(PyExc_AttributeError, name);
    return NULL;
}

static void set_single_parameter(const std::string& funcname, const std::string& paramname,
            const dtype& paramtype, char *metadata, char *data, const dtype& value)
{
    // TODO: Need dtype_dtype
    if (paramtype.get_type_id() != void_pointer_type_id) {
        stringstream ss;
        ss << "parameter \"" << paramname << "\" of dynd callable \"" << funcname << "\" with type " << paramtype;
        ss << " cannot accept a dtype as its value";
        throw runtime_error(ss.str());
    }
    *(const void **)data = value.extended() ? value.extended()
                                            : reinterpret_cast<const void *>(value.get_type_id());
}

static void set_single_parameter(const std::string& funcname, const std::string& paramname,
            const dtype& paramtype, char *metadata, char *data, const ndobject& value)
{
    // TODO: Need ndobject_dtype (but then we can get circular references, and need garbage collection :P)
    if (paramtype.get_type_id() != void_pointer_type_id) {
        stringstream ss;
        ss << "parameter \"" << paramname << "\" of dynd callable \"" << funcname << "\" with type " << paramtype;
        ss << " cannot accept a dtype as its value";
        throw runtime_error(ss.str());
    }
    *(const void **)data = value.get_ndo();
}

/**
 * This converts a single PyObject input parameter into the requested dynd
 * parameter data.
 */
static void set_single_parameter(const std::string& funcname, const std::string& paramname,
            const dtype& paramtype, char *metadata, char *data, PyObject *value)
{
    // Handle ndobject specially
    if (WNDObject_Check(value)) {
        if (paramtype.get_type_id() == void_pointer_type_id) {
            // Pass raw ndo pointers (with a borrowed reference) to void pointer params
            // TODO: Need ndobject_dtype (but then we can get circular references, and need garbage collection :P)
            *(const void **)data = ((WNDObject *)value)->v.get_ndo();
        } else {
            // Copy the value using the default mechanism
            const ndobject& n = ((WNDObject *)value)->v;
            dtype_assign(paramtype, metadata, data, n.get_dtype(), n.get_ndo_meta(), n.get_readonly_originptr());
        }
        return;
    }

    switch (paramtype.get_kind()) {
        case bool_kind:
            switch(paramtype.get_type_id()) {
                case bool_type_id: {
                    int result = PyObject_IsTrue(value);
                    if (result == -1) {
                        throw runtime_error("");
                    }
                    *reinterpret_cast<dynd_bool *>(data) = (result != 0);
                    return;
                }
                default:
                    break;
            }
            break;
        case int_kind: {
            pyobject_ownref ind(PyNumber_Index(value));
            PY_LONG_LONG ll = PyLong_AsLongLong(ind.get());
            if (ll == -1 && PyErr_Occurred()) {
                throw runtime_error("");
            }
            switch(paramtype.get_type_id()) {
                case int8_type_id: {
                    int8_t result = static_cast<int8_t>(ll);
                    if (result != ll) {
                        stringstream ss;
                        ss << "overflow passing value to parameter " << paramname << ", type " << paramtype << ", of function " << funcname;
                        throw runtime_error(ss.str());
                    }
                    *reinterpret_cast<int8_t *>(data) = result;
                    return;
                }
                case int16_type_id: {
                    int16_t result = static_cast<int16_t>(ll);
                    if (result != ll) {
                        stringstream ss;
                        ss << "overflow passing value to parameter " << paramname << ", type " << paramtype << ", of function " << funcname;
                        throw runtime_error(ss.str());
                    }
                    *reinterpret_cast<int16_t *>(data) = result;
                    return;
                }
                case int32_type_id: {
                    int32_t result = static_cast<int32_t>(ll);
                    if (result != ll) {
                        stringstream ss;
                        ss << "overflow passing value to parameter " << paramname << ", type " << paramtype << ", of function " << funcname;
                        throw runtime_error(ss.str());
                    }
                    *reinterpret_cast<int32_t *>(data) = result;
                    return;
                }
                case int64_type_id:
                    *reinterpret_cast<int64_t *>(data) = ll;
                    return;
            }
            break;
        }
        case uint_kind: {
            pyobject_ownref ind(PyNumber_Index(value));
            unsigned PY_LONG_LONG ull = PyLong_AsUnsignedLongLong(ind.get());
            if (ull == (unsigned PY_LONG_LONG)-1 && PyErr_Occurred()) {
                throw runtime_error("");
            }
            switch(paramtype.get_type_id()) {
                case uint8_type_id: {
                    uint8_t result = static_cast<uint8_t>(ull);
                    if (result != ull) {
                        stringstream ss;
                        ss << "overflow passing value to parameter " << paramname << ", type " << paramtype << ", of function " << funcname;
                        throw runtime_error(ss.str());
                    }
                    *reinterpret_cast<uint8_t *>(data) = result;
                    return;
                }
                case uint16_type_id: {
                    uint16_t result = static_cast<uint16_t>(ull);
                    if (result != ull) {
                        stringstream ss;
                        ss << "overflow passing value to parameter " << paramname << ", type " << paramtype << ", of function " << funcname;
                        throw runtime_error(ss.str());
                    }
                    *reinterpret_cast<uint16_t *>(data) = result;
                    return;
                }
                case uint32_type_id: {
                    uint32_t result = static_cast<uint32_t>(ull);
                    if (result != ull) {
                        stringstream ss;
                        ss << "overflow passing value to parameter " << paramname << ", type " << paramtype << ", of function " << funcname;
                        throw runtime_error(ss.str());
                    }
                    *reinterpret_cast<uint32_t *>(data) = result;
                    return;
                }
                case uint64_type_id:
                    *reinterpret_cast<uint64_t *>(data) = ull;
                    break;
            }
            break;
        }
        case real_kind: {
            double result = PyFloat_AsDouble(value);
            if (result == -1 && PyErr_Occurred()) {
                throw runtime_error("");
            }
            switch(paramtype.get_type_id()) {
                case float32_type_id:
                    *reinterpret_cast<float *>(data) = static_cast<float>(result);
                    return;
                case float64_type_id:
                    *reinterpret_cast<double *>(data) = result;
                    return;
            }
            break;
        }
        case complex_kind: {
            Py_complex result = PyComplex_AsCComplex(value);
            if (result.real == -1 && PyErr_Occurred()) {
                throw runtime_error("");
            }
            switch(paramtype.get_type_id()) {
                case complex_float32_type_id:
                    reinterpret_cast<float *>(data)[0] = static_cast<float>(result.real);
                    reinterpret_cast<float *>(data)[1] = static_cast<float>(result.imag);
                    return;
                case complex_float64_type_id:
                    reinterpret_cast<double *>(data)[0] = result.real;
                    reinterpret_cast<double *>(data)[1] = result.imag;
                    return;
            }
            break;
        }
        case string_kind: {
            const extended_string_dtype *esd = static_cast<const extended_string_dtype *>(paramtype.extended());
            string result = pystring_as_string(value);
            esd->set_utf8_string(metadata, data, assign_error_fractional, result);
            return;
        }
        default:
            break;
    }

    // Final, slow attempt to make it work, convert the input to an ndobject, then copy that value
    ndobject n = ndobject_from_py(value);
    dtype_assign(paramtype, metadata, data, n.get_dtype(), n.get_ndo_meta(), n.get_readonly_originptr());
}

PyObject *pydynd::call_gfunc_callable(const std::string& funcname, const dynd::gfunc::callable& c, const dtype& dt)
{
    const dtype& pdt = c.get_parameters_dtype();
    ndobject params(pdt);
    const fixedstruct_dtype *fsdt = static_cast<const fixedstruct_dtype *>(pdt.extended());
    if (fsdt->get_field_count() != 1) {
        stringstream ss;
        ss << "incorrect number of arguments for dynd callable \"" << funcname << "\" with parameters " << pdt;
        throw runtime_error(ss.str());
    }
    set_single_parameter(funcname, fsdt->get_field_names()[0], fsdt->get_field_types()[0],
            params.get_ndo_meta() + fsdt->get_metadata_offsets()[0],
            params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[0], dt);
    ndobject result = c.call_generic(params);
    if (result.get_dtype().is_scalar()) {
        return ndobject_as_py(result);
    } else {
        return wrap_ndobject(result);
    }
}

PyObject *pydynd::call_gfunc_callable(const std::string& funcname, const dynd::gfunc::callable& c, const dynd::ndobject& n)
{
    const dtype& pdt = c.get_parameters_dtype();
    ndobject params(pdt);
    const fixedstruct_dtype *fsdt = static_cast<const fixedstruct_dtype *>(pdt.extended());
    if (fsdt->get_field_count() != 1) {
        stringstream ss;
        ss << "not enough arguments for dynd callable \"" << funcname << "\" with parameters " << pdt;
        throw runtime_error(ss.str());
    }
    set_single_parameter(funcname, fsdt->get_field_names()[0], fsdt->get_field_types()[0],
            params.get_ndo_meta() + fsdt->get_metadata_offsets()[0],
            params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[0], n);
    return wrap_ndobject(c.call_generic(params));
}

PyObject *pydynd::wrap_ndobject_callable(const std::string& funcname, const dynd::gfunc::callable& c, const dynd::ndobject& n)
{
    WNDObjectCallable *result = (WNDObjectCallable *)WNDObjectCallable_Type->tp_alloc(WNDObjectCallable_Type, 0);
    if (!result) {
        return NULL;
    }
    // Calling tp_alloc doesn't call Cython's __cinit__, so do the placement new here
    placement_new(reinterpret_cast<pydynd::ndobject_callable_placement_wrapper &>(result->v));
    result->v.n = n;
    result->v.c = c;
    result->v.funcname = funcname;
    return (PyObject *)result;
}

PyObject *pydynd::ndobject_callable_call(const ndobject_callable_wrapper& ncw, PyObject *args, PyObject *kwargs)
{
    const dtype& pdt = ncw.c.get_parameters_dtype();
    ndobject params(pdt);
    const fixedstruct_dtype *fsdt = static_cast<const fixedstruct_dtype *>(pdt.extended());
    size_t param_count = fsdt->get_field_count() - 1, args_count = PyTuple_GET_SIZE(args);
    if (args_count > param_count) {
        stringstream ss;
        ss << "too many arguments for dynd callable \"" << ncw.funcname << "\" with parameters " << pdt;
        throw runtime_error(ss.str());
    }

    // Set the 'self' parameters
    set_single_parameter(ncw.funcname, fsdt->get_field_names()[0], fsdt->get_field_types()[0],
                params.get_ndo_meta() + fsdt->get_metadata_offsets()[0],
                params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[0], ncw.n);

    // Fill all the positional arguments
    for (size_t i = 0; i < args_count; ++i) {
        set_single_parameter(ncw.funcname, fsdt->get_field_names()[i+1], fsdt->get_field_types()[i+1],
                params.get_ndo_meta() + fsdt->get_metadata_offsets()[i+1],
                params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[i+1], PyTuple_GET_ITEM(args, i));
    }
    if (args_count == param_count) {
        if (kwargs != NULL) {
            Py_ssize_t size = PyDict_Size(kwargs);
            if (size != 0) {
                stringstream ss;
                ss << "too many arguments for dynd callable \"" << ncw.funcname << "\" with parameters " << pdt;
                throw runtime_error(ss.str());
            }
        }
    } else if (kwargs != NULL) {
        // Flags to make sure every parameter is filled (eventually would
        // use this to copy from default parameter values after filling kwargs)
        shortvector<char, 6> filled(param_count - args_count);
        memset(filled.get(), 0, param_count - args_count);
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            string s = pystring_as_string(key);
            size_t i;
            // Search for the parameter in the struct, and fill it if found
            for (i = args_count; i < param_count; ++i) {
                if (s == fsdt->get_field_names()[i+1]) {
                    set_single_parameter(ncw.funcname, fsdt->get_field_names()[i+1], fsdt->get_field_types()[i+1],
                            params.get_ndo_meta() + fsdt->get_metadata_offsets()[i+1],
                            params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[i+1], value);
                    filled[i - args_count] = 1;
                    break;
                }
            }
            if (i == param_count) {
                stringstream ss;
                ss << "dynd callable \"" << ncw.funcname << "\" with parameters " << pdt;
                ss << " does not have a parameter s";
                throw runtime_error(ss.str());
            }
        }
        // Check that all the arguments are full
        for (size_t i = 0; i < param_count - args_count; ++i) {
            if (filled[i] == 0) {
                stringstream ss;
                ss << "not enough arguments for dynd callable \"" << ncw.funcname << "\" with parameters " << pdt;
                throw runtime_error(ss.str());
            }
        }
    } else {
        stringstream ss;
        ss << "not enough arguments for dynd callable \"" << ncw.funcname << "\" with parameters " << pdt;
        throw runtime_error(ss.str());
    }

    return wrap_ndobject(ncw.c.call_generic(params));
}
