//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "gfunc_callable_functions.hpp"
#include "utility_functions.hpp"
#include "array_functions.hpp"
#include "array_as_py.hpp"
#include "array_assign_from_py.hpp"
#include "placement_wrappers.hpp"

#include <dynd/types/cstruct_type.hpp>
#include <dynd/types/builtin_type_properties.hpp>
#include <dynd/types/type_type.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

PyTypeObject *pydynd::WArrayCallable_Type;

void pydynd::init_w_array_callable_typeobject(PyObject *type)
{
    WArrayCallable_Type = (PyTypeObject *)type;
}

PyTypeObject *pydynd::WTypeCallable_Type;

void pydynd::init_w_ndt_type_callable_typeobject(PyObject *type)
{
    WTypeCallable_Type = (PyTypeObject *)type;
}


void pydynd::add_ndt_type_names_to_dir_dict(const ndt::type& dt, PyObject *dict)
{
    if (!dt.is_builtin()) {
        const std::pair<std::string, gfunc::callable> *properties;
        size_t count;
        // Add the type properties
        dt.extended()->get_dynamic_type_properties(&properties, &count);
        for (size_t i = 0; i < count; ++i) {
            if (PyDict_SetItemString(dict, properties[i].first.c_str(), Py_None) < 0) {
                throw runtime_error("");
            }
        }
        // Add the type functions
        dt.extended()->get_dynamic_type_functions(&properties, &count);
        for (size_t i = 0; i < count; ++i) {
            if (PyDict_SetItemString(dict, properties[i].first.c_str(), Py_None) < 0) {
                throw runtime_error("");
            }
        }
    }
}

PyObject *pydynd::get_ndt_type_dynamic_property(const dynd::ndt::type& dt, PyObject *name)
{
    if (!dt.is_builtin()) {
        const std::pair<std::string, gfunc::callable> *properties;
        size_t count;
        // Search for a property
        dt.extended()->get_dynamic_type_properties(&properties, &count);
        // TODO: We probably want to make some kind of acceleration structure for the name lookup
        if (count > 0) {
            string nstr = pystring_as_string(name);
            for (size_t i = 0; i < count; ++i) {
                if (properties[i].first == nstr) {
                    return call_gfunc_callable(nstr, properties[i].second, dt);
                }
            }
        }
        // Search for a function
        dt.extended()->get_dynamic_type_functions(&properties, &count);
        if (count > 0) {
            string nstr = pystring_as_string(name);
            for (size_t i = 0; i < count; ++i) {
                if (properties[i].first == nstr) {
                    return wrap_ndt_type_callable(nstr, properties[i].second, dt);
                }
            }
        }
    }

    PyErr_SetObject(PyExc_AttributeError, name);
    return NULL;
}

void pydynd::add_array_names_to_dir_dict(const dynd::nd::array& n, PyObject *dict)
{
    ndt::type dt = n.get_type();
    if (!dt.is_builtin()) {
        const std::pair<std::string, gfunc::callable> *properties;
        size_t count;
        // Add the array properties
        dt.extended()->get_dynamic_array_properties(&properties, &count);
        for (size_t i = 0; i < count; ++i) {
            if (PyDict_SetItemString(dict, properties[i].first.c_str(), Py_None) < 0) {
                throw runtime_error("");
            }
        }
        // Add the array functions
        dt.extended()->get_dynamic_array_functions(&properties, &count);
        for (size_t i = 0; i < count; ++i) {
            if (PyDict_SetItemString(dict, properties[i].first.c_str(), Py_None) < 0) {
                throw runtime_error("");
            }
        }
    } else {
        const std::pair<std::string, gfunc::callable> *properties;
        size_t count;
        // Add the array properties
        get_builtin_type_dynamic_array_properties(dt.get_type_id(), &properties, &count);
        for (size_t i = 0; i < count; ++i) {
            if (PyDict_SetItemString(dict, properties[i].first.c_str(), Py_None) < 0) {
                throw runtime_error("");
            }
        }
        // TODO: Add the array functions
    }
}

PyObject *pydynd::get_array_dynamic_property(const dynd::nd::array& n, PyObject *name)
{
    ndt::type dt = n.get_type();
    const std::pair<std::string, gfunc::callable> *properties;
    size_t count;
    // Search for a property
    if (!dt.is_builtin()) {
        dt.extended()->get_dynamic_array_properties(&properties, &count);
    } else {
        get_builtin_type_dynamic_array_properties(dt.get_type_id(), &properties, &count);
    }
    // TODO: We probably want to make some kind of acceleration structure for the name lookup
    if (count > 0) {
        string nstr = pystring_as_string(name);
        for (size_t i = 0; i < count; ++i) {
            if (properties[i].first == nstr) {
                return wrap_array(call_gfunc_callable(nstr, properties[i].second, n));
            }
        }
    }
    // Search for a function
    if (!dt.is_builtin()) {
        dt.extended()->get_dynamic_array_functions(&properties, &count);
    } else {
        count = 0;
    }
    if (count > 0) {
        string nstr = pystring_as_string(name);
        for (size_t i = 0; i < count; ++i) {
            if (properties[i].first == nstr) {
                return wrap_array_callable(nstr, properties[i].second, n);
            }
        }
    }

    PyErr_SetObject(PyExc_AttributeError, name);
    return NULL;
}

void pydynd::set_array_dynamic_property(const dynd::nd::array& n, PyObject *name, PyObject *value)
{
    ndt::type dt = n.get_type();
    const std::pair<std::string, gfunc::callable> *properties;
    size_t count;
    // Search for a property
    if (!dt.is_builtin()) {
        dt.extended()->get_dynamic_array_properties(&properties, &count);
    } else {
        get_builtin_type_dynamic_array_properties(dt.get_type_id(), &properties, &count);
    }
    // TODO: We probably want to make some kind of acceleration structure for the name lookup
    if (count > 0) {
        string nstr = pystring_as_string(name);
        for (size_t i = 0; i < count; ++i) {
            if (properties[i].first == nstr) {
                nd::array p = call_gfunc_callable(nstr, properties[i].second, n);
                array_broadcast_assign_from_py(p, value);
                return;
            }
        }
    }

    PyErr_SetObject(PyExc_AttributeError, name);
    throw exception();
}

static void set_single_parameter(const std::string& funcname, const std::string& paramname,
            const ndt::type& paramtype, char *metadata, char *data, const ndt::type& value)
{
    if (paramtype.get_type_id() != type_type_id) {
        stringstream ss;
        ss << "parameter \"" << paramname << "\" of dynd callable \"" << funcname << "\" with type " << paramtype;
        ss << " cannot accept a dynd type as its value";
        throw runtime_error(ss.str());
    }
    // The type is encoded as either a raw type id, or a pointer to an base_type,
    // just as the gfunc object is expecting.
    ndt::type(value).swap(reinterpret_cast<type_type_data *>(data)->dt);
}

static void set_single_parameter(const std::string& funcname, const std::string& paramname,
            const ndt::type& paramtype, char *metadata, char *data, const nd::array& value)
{
    // TODO: Need array_type (but then we can get circular references, and need garbage collection :P)
    if (paramtype.get_type_id() != void_pointer_type_id) {
        stringstream ss;
        ss << "parameter \"" << paramname << "\" of dynd callable \"" << funcname << "\" with type " << paramtype;
        ss << " cannot accept an array as its value";
        throw runtime_error(ss.str());
    }
    *(const void **)data = value.get_ndo();
}

/**
 * This converts a single PyObject input parameter into the requested dynd
 * parameter data.
 *
 * \param out_storage  This is a hack because dynd doesn't support object lifetime management
 */
static void set_single_parameter(const ndt::type& paramtype, char *metadata, char *data,
                PyObject *value, vector<nd::array>& out_storage)
{
    if (paramtype.get_type_id() == void_pointer_type_id) {
        out_storage.push_back(array_from_py(value, 0, false));
        // TODO: Need array_type (but then we can get circular references, and need garbage collection :P)
        *(const void **)data = out_storage.back().get_ndo();
    } else {
        array_nodim_broadcast_assign_from_py(paramtype, metadata, data, value);
    }
}

PyObject *pydynd::call_gfunc_callable(const std::string& funcname, const dynd::gfunc::callable& c, const ndt::type& dt)
{
    const ndt::type& pdt = c.get_parameters_type();
    nd::array params = nd::empty(pdt);
    const cstruct_type *fsdt = static_cast<const cstruct_type *>(pdt.extended());
    if (fsdt->get_field_count() != 1) {
        stringstream ss;
        ss << "incorrect number of arguments for dynd callable \"" << funcname << "\" with parameters " << pdt;
        throw runtime_error(ss.str());
    }
    set_single_parameter(funcname, fsdt->get_field_names()[0], fsdt->get_field_types()[0],
            params.get_ndo_meta() + fsdt->get_metadata_offsets()[0],
            params.get_ndo()->m_data_pointer + fsdt->get_data_offsets_vector()[0], dt);
    nd::array result = c.call_generic(params);
    if (result.get_type().is_scalar()) {
        return array_as_py(result);
    } else {
        return wrap_array(result);
    }
}

nd::array pydynd::call_gfunc_callable(const std::string& funcname, const dynd::gfunc::callable& c, const dynd::nd::array& n)
{
    const ndt::type& pdt = c.get_parameters_type();
    nd::array params = nd::empty(pdt);
    const cstruct_type *fsdt = static_cast<const cstruct_type *>(pdt.extended());
    if (fsdt->get_field_count() != 1) {
        stringstream ss;
        ss << "not enough arguments for dynd callable \"" << funcname << "\" with parameters " << pdt;
        throw runtime_error(ss.str());
    }
    set_single_parameter(funcname, fsdt->get_field_names()[0], fsdt->get_field_types()[0],
            params.get_ndo_meta() + fsdt->get_metadata_offsets()[0],
            params.get_ndo()->m_data_pointer + fsdt->get_data_offsets_vector()[0], n);
    return c.call_generic(params);
}

/**
 * Fills all the parameters after the first one from the args/kwargs.
 *
 * \param out_storage  This is a hack because dynd doesn't support object lifetime management
 */
static void fill_thiscall_parameters_array(const string& funcname, const gfunc::callable &c, PyObject *args, PyObject *kwargs,
                nd::array& out_params, vector<nd::array>& out_storage)
{
    const ndt::type& pdt = c.get_parameters_type();
    const cstruct_type *fsdt = static_cast<const cstruct_type *>(pdt.extended());
    size_t param_count = fsdt->get_field_count() - 1, args_count = PyTuple_GET_SIZE(args);
    if (args_count > param_count) {
        stringstream ss;
        ss << "too many arguments for dynd callable \"" << funcname << "\" with parameters " << pdt;
        throw runtime_error(ss.str());
    }

    // Fill all the positional arguments
    for (size_t i = 0; i < args_count; ++i) {
        set_single_parameter(fsdt->get_field_types()[i+1],
                out_params.get_ndo_meta() + fsdt->get_metadata_offsets()[i+1],
                out_params.get_ndo()->m_data_pointer + fsdt->get_data_offsets_vector()[i+1],
                PyTuple_GET_ITEM(args, i), out_storage);
    }

    // Fill in the keyword arguments if any are provided
    if (kwargs != NULL && PyDict_Size(kwargs) > 0) {
        // Make sure there aren't too many arguments
        if (args_count == param_count) {
            stringstream ss;
            ss << "too many arguments for dynd callable \"" << funcname << "\" with parameters " << pdt;
            throw runtime_error(ss.str());
        }

        // Flags to make sure every parameter is filled
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
                    set_single_parameter(fsdt->get_field_types()[i+1],
                            out_params.get_ndo_meta() + fsdt->get_metadata_offsets()[i+1],
                            out_params.get_ndo()->m_data_pointer + fsdt->get_data_offsets_vector()[i+1], value, out_storage);
                    filled[i - args_count] = 1;
                    break;
                }
            }
            if (i == param_count) {
                stringstream ss;
                ss << "dynd callable \"" << funcname << "\" with parameters " << pdt;
                ss << " does not have a parameter " << s;
                throw runtime_error(ss.str());
            }
        }
        // Fill in missing parameters from the defaults
        const nd::array& default_parameters = c.get_default_parameters();
        if (!default_parameters.is_empty()) {
            // Figure out where to start filling in default parameters
            int first_default_param = c.get_first_default_parameter() - 1;
            if (first_default_param < (int)param_count) {
                if (first_default_param < (int)args_count) {
                    first_default_param = (int)args_count;
                }
                for (size_t i = first_default_param; i < param_count; ++i) {
                    // Fill in the parameters which haven't been touched yet
                    if (filled[i - args_count] == 0) {
                        size_t metadata_offset = fsdt->get_metadata_offsets()[i+1];
                        size_t data_offset = fsdt->get_data_offsets_vector()[i+1];
                        typed_data_copy(fsdt->get_field_types()[i+1],
                                        out_params.get_ndo_meta() + metadata_offset,
                                        out_params.get_ndo()->m_data_pointer + data_offset,
                                        default_parameters.get_ndo_meta() + metadata_offset,
                                        default_parameters.get_ndo()->m_data_pointer + data_offset);
                        filled[i - args_count] = 1;
                    }
                }
            }
        }
        // Check that all the arguments are full
        for (size_t i = 0; i < param_count - args_count; ++i) {
            if (filled[i] == 0) {
                stringstream ss;
                ss << "not enough arguments for dynd callable \"" << funcname << "\" with parameters " << pdt;
                throw runtime_error(ss.str());
            }
        }
    } else if (args_count < param_count) {
        // Fill in missing parameters from the defaults
        const nd::array& default_parameters = c.get_default_parameters();
        if (!default_parameters.is_empty()) {
            // Figure out where to start filling in default parameters
            int first_default_param = c.get_first_default_parameter() - 1;
            if (first_default_param < (int)param_count && first_default_param <= (int)args_count) {
                for (size_t i = args_count; i < param_count; ++i) {
                    size_t metadata_offset = fsdt->get_metadata_offsets()[i+1];
                    size_t data_offset = fsdt->get_data_offsets_vector()[i+1];
                    typed_data_copy(fsdt->get_field_types()[i+1],
                                    out_params.get_ndo_meta() + metadata_offset,
                                    out_params.get_ndo()->m_data_pointer + data_offset,
                                    default_parameters.get_ndo_meta() + metadata_offset,
                                    default_parameters.get_ndo()->m_data_pointer + data_offset);
                }
            } else {
                stringstream ss;
                ss << "not enough arguments for dynd callable \"" << funcname << "\" with parameters " << pdt;
                throw runtime_error(ss.str());
            }
        } else {
            stringstream ss;
            ss << "not enough arguments for dynd callable \"" << funcname << "\" with parameters " << pdt;
            throw runtime_error(ss.str());
        }
    }
}

PyObject *pydynd::wrap_array_callable(const std::string& funcname, const dynd::gfunc::callable& c, const dynd::nd::array& n)
{
    WArrayCallable *result = (WArrayCallable *)WArrayCallable_Type->tp_alloc(WArrayCallable_Type, 0);
    if (!result) {
        return NULL;
    }
    // Calling tp_alloc doesn't call Cython's __cinit__, so do the placement new here
    placement_new(reinterpret_cast<pydynd::array_callable_placement_wrapper &>(result->v));
    result->v.n = n;
    result->v.c = c;
    result->v.funcname = funcname;
    return (PyObject *)result;
}

PyObject *pydynd::array_callable_call(const array_callable_wrapper& ncw, PyObject *args, PyObject *kwargs)
{
    const ndt::type& pdt = ncw.c.get_parameters_type();
    vector<nd::array> storage;
    nd::array params = nd::empty(pdt);
    const cstruct_type *fsdt = static_cast<const cstruct_type *>(pdt.extended());
    // Set the 'self' parameter value
    set_single_parameter(ncw.funcname, fsdt->get_field_names()[0], fsdt->get_field_types()[0],
                params.get_ndo_meta() + fsdt->get_metadata_offsets()[0],
                params.get_ndo()->m_data_pointer + fsdt->get_data_offsets_vector()[0], ncw.n);

    fill_thiscall_parameters_array(ncw.funcname, ncw.c, args, kwargs, params, storage);

    return wrap_array(ncw.c.call_generic(params));
}

PyObject *pydynd::wrap_ndt_type_callable(const std::string& funcname, const dynd::gfunc::callable& c, const dynd::ndt::type& d)
{
    WTypeCallable *result = (WTypeCallable *)WTypeCallable_Type->tp_alloc(WTypeCallable_Type, 0);
    if (!result) {
        return NULL;
    }
    // Calling tp_alloc doesn't call Cython's __cinit__, so do the placement new here
    placement_new(reinterpret_cast<pydynd::ndt_type_callable_placement_wrapper &>(result->v));
    result->v.d = d;
    result->v.c = c;
    result->v.funcname = funcname;
    return (PyObject *)result;
}

static PyObject *ndt_type_callable_call(const std::string& funcname, const gfunc::callable& c,
                const ndt::type& d, PyObject *args, PyObject *kwargs)
{
    const ndt::type& pdt = c.get_parameters_type();
    vector<nd::array> storage;
    nd::array params = nd::empty(pdt);
    const cstruct_type *fsdt = static_cast<const cstruct_type *>(pdt.extended());
    // Set the 'self' parameter value
    set_single_parameter(funcname, fsdt->get_field_names()[0], fsdt->get_field_types()[0],
                params.get_ndo_meta() + fsdt->get_metadata_offsets()[0],
                params.get_ndo()->m_data_pointer + fsdt->get_data_offsets_vector()[0], d);

    fill_thiscall_parameters_array(funcname, c, args, kwargs, params, storage);

    return wrap_array(c.call_generic(params));
}

PyObject *pydynd::ndt_type_callable_call(const ndt_type_callable_wrapper& dcw, PyObject *args, PyObject *kwargs)
{
    return ndt_type_callable_call(dcw.funcname, dcw.c, dcw.d, args, kwargs);
}

PyObject *pydynd::call_ndt_type_constructor_function(const dynd::ndt::type& dt, PyObject *args, PyObject *kwargs)
{
    // First find the __construct__ callable
    if (!dt.is_builtin()) {
        const std::pair<std::string, gfunc::callable> *properties;
        size_t count;
        // Search for a function
        dt.extended()->get_dynamic_type_functions(&properties, &count);
        if (count > 0) {
            for (size_t i = 0; i < count; ++i) {
                if (properties[i].first == "__construct__") {
                    return ndt_type_callable_call("__construct__", properties[i].second, dt, args, kwargs);
                }
            }
        }
    }

    stringstream ss;
    ss << "dynd type " << dt << " has no array constructor function";
    PyErr_SetString(PyExc_TypeError, ss.str().c_str());
    return NULL;
}
