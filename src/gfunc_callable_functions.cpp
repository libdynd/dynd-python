//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "gfunc_callable_functions.hpp"
#include "utility_functions.hpp"
#include "ndobject_functions.hpp"
#include "ndobject_as_py.hpp"

#include <dynd/dtypes/fixedstruct_dtype.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

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
        dt.extended()->get_dynamic_ndobject_properties(&properties, &count);
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

PyObject *pydynd::call_gfunc_callable(const std::string& funcname, const dynd::gfunc::callable& c, const dtype& dt)
{
    const dtype& pdt = c.get_parameters_dtype();
    ndobject params(pdt);
    const fixedstruct_dtype *fsdt = static_cast<const fixedstruct_dtype *>(pdt.extended());
    if (fsdt->get_field_types().size() != 1) {
        stringstream ss;
        ss << "not enough arguments for dynd callable \"" << funcname << "\" with parameters " << pdt;
        throw runtime_error(ss.str());
    }
    set_single_parameter(funcname, fsdt->get_field_names()[0], fsdt->get_field_types()[0],
            params.get_ndo_meta() + fsdt->get_metadata_offsets()[0],
            params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[0], dt);
    ndobject result = c.call(params);
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
    if (fsdt->get_field_types().size() != 1) {
        stringstream ss;
        ss << "not enough arguments for dynd callable \"" << funcname << "\" with parameters " << pdt;
        throw runtime_error(ss.str());
    }
    set_single_parameter(funcname, fsdt->get_field_names()[0], fsdt->get_field_types()[0],
            params.get_ndo_meta() + fsdt->get_metadata_offsets()[0],
            params.get_ndo()->m_data_pointer + fsdt->get_data_offsets()[0], n);
    return wrap_ndobject(c.call(params));
}