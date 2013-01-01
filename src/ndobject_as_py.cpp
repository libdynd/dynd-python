//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>
#include <datetime.h>

#include "ndobject_as_py.hpp"
#include "ndobject_functions.hpp"
#include "utility_functions.hpp"

#include <dynd/dtypes/strided_array_dtype.hpp>
#include <dynd/dtypes/date_dtype.hpp>

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

static PyObject* element_as_pyobject(const dtype& d, const char *data, const char *metadata)
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
            return PyInt_FromLong(*(const int8_t *)data);
        case int16_type_id:
            return PyInt_FromLong(*(const int16_t *)data);
        case int32_type_id:
            return PyInt_FromLong(*(const int32_t *)data);
        case int64_type_id:
            return PyLong_FromLongLong(*(const int64_t *)data);
        case uint8_type_id:
            return PyInt_FromLong(*(const uint8_t *)data);
        case uint16_type_id:
            return PyInt_FromLong(*(const uint16_t *)data);
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
        case fixedstring_type_id: {
            const base_string_dtype *esd = static_cast<const base_string_dtype *>(d.extended());
            switch (esd->get_encoding()) {
                case string_encoding_ascii:
                    return PyUnicode_DecodeASCII(data, strnlen(data, d.get_data_size()), NULL);
                case string_encoding_utf_8:
                    return PyUnicode_DecodeUTF8(data, strnlen(data, d.get_data_size()), NULL);
                case string_encoding_ucs_2:
                case string_encoding_utf_16: {
                    // Get the null-terminated string length
                    const uint16_t *udata = (const uint16_t *)data;
                    const uint16_t *udata_end = udata;
                    intptr_t size = d.get_data_size() / sizeof(uint16_t);
                    while (size > 0 && *udata_end != 0) {
                        --size;
                        ++udata_end;
                    }
                    return PyUnicode_DecodeUTF16(data, sizeof(uint16_t) * (udata_end - udata), NULL, NULL);
                }
                case string_encoding_utf_32: {
                    // Get the null-terminated string length
                    const uint32_t *udata = (const uint32_t *)data;
                    const uint32_t *udata_end = udata;
                    intptr_t size = d.get_data_size() / sizeof(uint32_t);
                    while (size > 0 && *udata_end != 0) {
                        --size;
                        ++udata_end;
                    }
                    return PyUnicode_DecodeUTF32(data, sizeof(uint32_t) * (udata_end - udata), NULL, NULL);
                }
                default:
                    throw runtime_error("Unrecognized dynd::ndobject string encoding");
            }
        }
        case string_type_id: {
            const char * const *refs = reinterpret_cast<const char * const *>(data);
            const base_string_dtype *esd = static_cast<const base_string_dtype *>(d.extended());
            switch (esd->get_encoding()) {
                case string_encoding_ascii:
                    return PyUnicode_DecodeASCII(refs[0], refs[1] - refs[0], NULL);
                case string_encoding_utf_8:
                    return PyUnicode_DecodeUTF8(refs[0], refs[1] - refs[0], NULL);
                case string_encoding_ucs_2:
                case string_encoding_utf_16:
                    return PyUnicode_DecodeUTF16(refs[0], refs[1] - refs[0], NULL, NULL);
                case string_encoding_utf_32:
                    return PyUnicode_DecodeUTF32(refs[0], refs[1] - refs[0], NULL, NULL);
                default:
                    throw runtime_error("Unrecognized dynd::ndobject string encoding");
            }
        }
        case date_type_id: {
            const date_dtype *dd = static_cast<const date_dtype *>(d.extended());
            int32_t year, month, day;
            dd->get_ymd(metadata, data, year, month, day);
            return PyDate_FromDate(year, month, day);
        }
        default: {
            stringstream ss;
            ss << "Cannot convert dynd::ndobject with dtype " << d << " into python object";
            throw runtime_error(ss.str());
        }
    }
}

namespace {
    struct ndobject_as_py_data {
        pyobject_ownref result;
        int index;
    };

    static void nested_ndobject_as_py(const dtype& d, char *data, const char *metadata, void *result)
    {
        ndobject_as_py_data *r = reinterpret_cast<ndobject_as_py_data *>(result);

        ndobject_as_py_data el;
        if (d.is_scalar()) {
            el.result.reset(element_as_pyobject(d, data, metadata));
        } else {
            intptr_t size = d.get_dim_size(data, metadata);
            el.result.reset(PyList_New(size));
            el.index = 0;

            d.extended()->foreach_leading(data, metadata, &nested_ndobject_as_py, &el);
        }

        if (r->result) {
            PyList_SET_ITEM(r->result.get(), r->index++, el.result.release());
        } else {
            r->result.reset(el.result.release());
        }
    }
} // anonymous namespace

PyObject* pydynd::ndobject_as_py(const dynd::ndobject& n)
{
    // Evaluate the ndobject
    ndobject nvals = n.vals();
    ndobject_as_py_data result;

    nested_ndobject_as_py(nvals.get_dtype(), nvals.get_ndo()->m_data_pointer, nvals.get_ndo_meta(), &result);
    return result.result.release();
}

