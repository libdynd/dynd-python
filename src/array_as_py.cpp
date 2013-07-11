//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>
#include <datetime.h>

#include "array_as_py.hpp"
#include "array_functions.hpp"
#include "type_functions.hpp"
#include "utility_functions.hpp"

#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/type_type.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

// Initialize the pydatetime API
namespace {
struct init_pydatetime {
    init_pydatetime() {
        PyDateTime_IMPORT;
    }
};
init_pydatetime pdt;
} // anonymous namespace

static PyObject* element_as_pyobject(const ndt::type& d, const char *data, const char *metadata)
{
    switch (d.get_type_id()) {
        case bool_type_id:
            if (*(const dynd_bool *)data) {
                Py_INCREF(Py_True);
                return Py_True;
            } else {
                Py_INCREF(Py_False);
                return Py_False;
            }
        case int8_type_id:
#if PY_VERSION_HEX >= 0x03000000
            return PyLong_FromLong(*(const int8_t *)data);
#else
            return PyInt_FromLong(*(const int8_t *)data);
#endif
        case int16_type_id:
#if PY_VERSION_HEX >= 0x03000000
            return PyLong_FromLong(*(const int16_t *)data);
#else
            return PyInt_FromLong(*(const int16_t *)data);
#endif
        case int32_type_id:
#if PY_VERSION_HEX >= 0x03000000
            return PyLong_FromLong(*(const int32_t *)data);
#else
            return PyInt_FromLong(*(const int32_t *)data);
#endif
        case int64_type_id:
            return PyLong_FromLongLong(*(const int64_t *)data);
        case uint8_type_id:
#if PY_VERSION_HEX >= 0x03000000
            return PyLong_FromLong(*(const uint8_t *)data);
#else
            return PyInt_FromLong(*(const uint8_t *)data);
#endif
        case uint16_type_id:
#if PY_VERSION_HEX >= 0x03000000
            return PyLong_FromLong(*(const uint16_t *)data);
#else
            return PyInt_FromLong(*(const uint16_t *)data);
#endif
        case uint32_type_id:
            return PyLong_FromUnsignedLong(*(const uint32_t *)data);
        case uint64_type_id:
            return PyLong_FromUnsignedLongLong(*(const uint64_t *)data);
        case float32_type_id:
            return PyFloat_FromDouble(*(const float *)data);
        case float64_type_id:
            return PyFloat_FromDouble(*(const double *)data);
        case complex_float32_type_id:
            return PyComplex_FromDoubles(*(const float *)data, *((const float *)data + 1));
        case complex_float64_type_id:
            return PyComplex_FromDoubles(*(const double *)data, *((const double *)data + 1));
        case fixedbytes_type_id:
            return PyBytes_FromStringAndSize(data, d.get_data_size());
        case bytes_type_id: {
            const bytes_type_data *d = reinterpret_cast<const bytes_type_data *>(data);
            return PyBytes_FromStringAndSize(d->begin, d->end - d->begin);
        }
        case fixedstring_type_id:
        case string_type_id:
        case json_type_id: {
            const char *begin = NULL, *end = NULL;
            const base_string_type *esd = static_cast<const base_string_type *>(d.extended());
            esd->get_string_range(&begin, &end, metadata, data);
            switch (esd->get_encoding()) {
                case string_encoding_ascii:
                    return PyUnicode_DecodeASCII(begin, end - begin, NULL);
                case string_encoding_utf_8:
                    return PyUnicode_DecodeUTF8(begin, end - begin, NULL);
                case string_encoding_ucs_2:
                case string_encoding_utf_16:
                    return PyUnicode_DecodeUTF16(begin, end - begin, NULL, NULL);
                case string_encoding_utf_32:
                    return PyUnicode_DecodeUTF32(begin, end - begin, NULL, NULL);
                default:
                    throw runtime_error("Unrecognized dynd array string encoding");
            }
        }
        case date_type_id: {
            const date_type *dd = static_cast<const date_type *>(d.extended());
            int32_t year, month, day;
            dd->get_ymd(metadata, data, year, month, day);
            return PyDate_FromDate(year, month, day);
        }
        case datetime_type_id: {
            const datetime_type *dd = static_cast<const datetime_type *>(d.extended());
            int32_t year, month, day, hour, minute, second, nsecond;
            dd->get_cal(metadata, data, year, month, day, hour, minute, second, nsecond);
            int32_t usecond = nsecond / 1000;
            if (usecond * 1000 != nsecond) {
                stringstream ss;
                ss << "cannot convert dynd value of type " << d;
                ss << " with value ";
                dd->print_data(ss, metadata, data);
                ss << " to python without precision loss";
                throw runtime_error(ss.str());
            }
            return PyDateTime_FromDateAndTime(year, month, day, hour, minute, second, usecond);
        }
        case type_type_id: {
            ndt::type dt(reinterpret_cast<const type_type_data *>(data)->dt, true);
            return wrap_ndt_type(DYND_MOVE(dt));
        }
        default: {
            stringstream ss;
            ss << "Cannot convert dynd array with dtype " << d << " into python object";
            throw runtime_error(ss.str());
        }
    }
}

namespace {
    struct array_as_py_data {
        pyobject_ownref result;
        int index;
    };
} // anonymous namespace

static void nested_array_as_py(const ndt::type& d, char *data, const char *metadata, void *result);

static void nested_struct_as_py(const ndt::type& d, char *data, const char *metadata, void *result)
{
    array_as_py_data *r = reinterpret_cast<array_as_py_data *>(result);

    const base_struct_type *bsd = static_cast<const base_struct_type *>(d.extended());
    size_t field_count = bsd->get_field_count();
    const string *field_names = bsd->get_field_names();
    const ndt::type *field_types = bsd->get_field_types();
    const size_t *field_metadata_offsets = bsd->get_metadata_offsets();
    const size_t *field_data_offsets = bsd->get_data_offsets(metadata);

    r->result.reset(PyDict_New());
    for (size_t i = 0; i != field_count; ++i) {
        const string& fname = field_names[i];
        pyobject_ownref key(PyUnicode_DecodeUTF8(fname.data(), fname.size(), NULL));
        array_as_py_data temp_el;
        nested_array_as_py(field_types[i], data + field_data_offsets[i],
                        metadata + field_metadata_offsets[i], &temp_el);
        if (PyDict_SetItem(r->result.get(), key.get(), temp_el.result.get()) < 0) {
            throw runtime_error("propagating dict setitem error");
        }
    }
}

static void nested_array_as_py(const ndt::type& d, char *data, const char *metadata, void *result)
{
    array_as_py_data *r = reinterpret_cast<array_as_py_data *>(result);

    array_as_py_data el;
    if (d.is_scalar()) {
        el.result.reset(element_as_pyobject(d, data, metadata));
    } else if (d.get_kind() == struct_kind) {
        nested_struct_as_py(d, data, metadata, &el);
    } else {
        intptr_t size = d.get_dim_size(metadata, data);
        el.result.reset(PyList_New(size));
        el.index = 0;

        d.extended()->foreach_leading(data, metadata, &nested_array_as_py, &el);
    }

    if (r->result) {
        PyList_SET_ITEM(r->result.get(), r->index++, el.result.release());
    } else {
        r->result.reset(el.result.release());
    }
}

PyObject* pydynd::array_as_py(const dynd::nd::array& n)
{
    // Evaluate the nd::array
    nd::array nvals = n.eval();
    array_as_py_data result;

    nested_array_as_py(nvals.get_type(), nvals.get_ndo()->m_data_pointer, nvals.get_ndo_meta(), &result);
    return result.result.release();
}

