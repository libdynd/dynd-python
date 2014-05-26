//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
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
#include <dynd/types/time_type.hpp>
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

PyObject *pydynd::pylong_from_uint128(const dynd_uint128& val)
{
    if (val.m_hi == 0ULL) {
        return PyLong_FromUnsignedLongLong(val.m_lo);
    }
    // Use the pynumber methods to shift and or together the 64 bit parts
    pyobject_ownref hi(PyLong_FromUnsignedLongLong(val.m_hi));
    pyobject_ownref sixtyfour(PyLong_FromLong(64));
    pyobject_ownref hi_shifted(PyNumber_Lshift(hi.get(), sixtyfour));
    pyobject_ownref lo(PyLong_FromUnsignedLongLong(val.m_lo));
    return PyNumber_Or(hi_shifted.get(), lo.get());
}

PyObject *pydynd::pylong_from_int128(const dynd_int128& val)
{
    if (val.is_negative()) {
        if (val.m_hi == 0xffffffffffffffffULL &&
                (val.m_hi & 0x8000000000000000ULL) != 0) {
            return PyLong_FromLongLong(static_cast<int64_t>(val.m_lo));
        }
        pyobject_ownref absval(
            pylong_from_uint128(static_cast<dynd_uint128>(-val)));
        return PyNumber_Negative(absval.get());
    } else {
        return pylong_from_uint128(static_cast<dynd_uint128>(val));
    }
}

/**
 * Converts an array element into a python object.
 *
 * \param d  The type of the data.
 * \param metadata  The arrmeta of the data.
 * \param data  The data pointer.
 */
static PyObject *element_as_pyobject(const ndt::type &d, const char *metadata,
                                     const char *data)
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
        case int128_type_id:
            return pylong_from_int128(*(const dynd_int128 *)data);
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
        case uint128_type_id:
            return pylong_from_uint128(*(const dynd_uint128 *)data);
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
            const base_string_type *esd = d.tcast<base_string_type>();
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
                    throw dynd::type_error("Unrecognized dynd array string encoding");
            }
        }
        case date_type_id: {
            const date_type *dd = d.tcast<date_type>();
            date_ymd ymd = dd->get_ymd(metadata, data);
            return PyDate_FromDate(ymd.year, ymd.month, ymd.day);
        }
        case time_type_id: {
            const time_type *tt = d.tcast<time_type>();
            time_hmst hmst = tt->get_time(metadata, data);
            return PyTime_FromTime(hmst.hour, hmst.minute, hmst.second,
                                   hmst.tick / DYND_TICKS_PER_MICROSECOND);
        }
        case datetime_type_id: {
            const datetime_type *dd = d.tcast<datetime_type>();
            int32_t year, month, day, hour, minute, second, tick;
            dd->get_cal(metadata, data, year, month, day, hour, minute, second, tick);
            int32_t usecond = tick / 10;
            return PyDateTime_FromDateAndTime(year, month, day, hour, minute, second, usecond);
        }
        case type_type_id: {
            ndt::type tp(reinterpret_cast<const type_type_data *>(data)->tp, true);
            return wrap_ndt_type(DYND_MOVE(tp));
        }
        default: {
            stringstream ss;
            ss << "Cannot convert dynd array with dtype " << d << " into python object";
            throw dynd::type_error(ss.str());
        }
    }
}

namespace {
    struct array_as_py_data {
        pyobject_ownref result;
        int index;
        /** When this is true, structs are converted to tuples instead of dicts
         */
        bool struct_as_pytuple;
    };
} // anonymous namespace

static void nested_array_as_py(const ndt::type& d, const char *metadata, char *data, void *result);

static void nested_struct_as_py(const ndt::type& d, const char *metadata, char *data, void *result)
{
    array_as_py_data *r = reinterpret_cast<array_as_py_data *>(result);

    const base_struct_type *bsd = d.tcast<base_struct_type>();
    size_t field_count = bsd->get_field_count();
    const uintptr_t *field_arrmeta_offsets = bsd->get_arrmeta_offsets_raw();
    const uintptr_t *field_data_offsets = bsd->get_data_offsets(metadata);

    if (r->struct_as_pytuple) {
        r->result.reset(PyTuple_New(field_count));
        for (size_t i = 0; i != field_count; ++i) {
            array_as_py_data temp_el;
            temp_el.struct_as_pytuple = r->struct_as_pytuple;
            nested_array_as_py(bsd->get_field_type(i), metadata + field_arrmeta_offsets[i],
                               data + field_data_offsets[i], &temp_el);
            PyTuple_SET_ITEM(r->result.get(), i, temp_el.result.release());
        }
    } else {
        r->result.reset(PyDict_New());
        for (size_t i = 0; i != field_count; ++i) {
            const string_type_data& fname = bsd->get_field_name_raw(i);
            pyobject_ownref key(PyUnicode_DecodeUTF8(fname.begin, fname.end - fname.begin, NULL));
            array_as_py_data temp_el;
            temp_el.struct_as_pytuple = r->struct_as_pytuple;
            nested_array_as_py(bsd->get_field_type(i), metadata + field_arrmeta_offsets[i],
                               data + field_data_offsets[i], &temp_el);
            if (PyDict_SetItem(r->result.get(), key.get(), temp_el.result.get()) < 0) {
                throw runtime_error("propagating dict setitem error");
            }
        }
    }
}

/**
 * Converts an array element into a python object.
 *
 * \param d  The type of the data.
 * \param metadata  The arrmeta of the data.
 * \param data  The data pointer.
 * \param result An array_as_py_data struct used to track the result and
 *               carry parameters through the foreach_leading.
 */
static void nested_array_as_py(const ndt::type &d, const char *metadata,
                               char *data, void *result)
{
    array_as_py_data *r = reinterpret_cast<array_as_py_data *>(result);

    array_as_py_data el;
    el.struct_as_pytuple = r->struct_as_pytuple;
    if (d.is_scalar()) {
        el.result.reset(element_as_pyobject(d, metadata, data));
    } else if (d.get_kind() == struct_kind) {
        nested_struct_as_py(d, metadata, data, &el);
    } else {
        intptr_t size = d.get_dim_size(metadata, data);
        el.result.reset(PyList_New(size));
        el.index = 0;

        d.extended()->foreach_leading(metadata, data, &nested_array_as_py, &el);
    }

    if (r->result) {
        PyList_SET_ITEM(r->result.get(), r->index++, el.result.release());
    } else {
        r->result.reset(el.result.release());
    }
}

PyObject* pydynd::array_as_py(const dynd::nd::array& n, bool struct_as_pytuple)
{
    // Evaluate the nd::array
    nd::array nvals = n.eval();
    array_as_py_data result;
    result.struct_as_pytuple = struct_as_pytuple;

    nested_array_as_py(nvals.get_type(), nvals.get_arrmeta(), nvals.get_ndo()->m_data_pointer, &result);
    return result.result.release();
}

