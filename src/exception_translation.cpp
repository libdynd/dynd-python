//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include <stdexcept>

#include <dynd/exceptions.hpp>

#include "exception_translation.hpp"

namespace {

    PyObject *BroadcastException = NULL;
} // anonymous namespace

void pydynd::translate_exception()
{
    try {
        if (PyErr_Occurred())
            ; // let the latest Python exn pass through and ignore the current one
        else
            throw;
    } catch (const dynd::broadcast_error& exn) {
        PyErr_SetString(BroadcastException, exn.message());
    } catch (const dynd::too_many_indices& exn) {
        PyErr_SetString(PyExc_IndexError, exn.message());
    } catch (const dynd::index_out_of_bounds& exn) {
        PyErr_SetString(PyExc_IndexError, exn.message());
    } catch (const dynd::axis_out_of_bounds& exn) {
        PyErr_SetString(PyExc_IndexError, exn.message());
    } catch (const dynd::irange_out_of_bounds& exn) {
        PyErr_SetString(PyExc_IndexError, exn.message());
    } catch (const dynd::invalid_type_id& exn) {
        PyErr_SetString(PyExc_TypeError, exn.message());
    } catch (const std::bad_alloc& exn) {
        PyErr_SetString(PyExc_MemoryError, exn.what());
    } catch (const std::bad_cast& exn) {
        PyErr_SetString(PyExc_TypeError, exn.what());
    } catch (const std::domain_error& exn) {
        PyErr_SetString(PyExc_ValueError, exn.what());
    } catch (const std::invalid_argument& exn) {
        PyErr_SetString(PyExc_ValueError, exn.what());
    } catch (const std::out_of_range& exn) {
        PyErr_SetString(PyExc_IndexError, exn.what());
    } catch (const std::overflow_error& exn) {
        PyErr_SetString(PyExc_OverflowError, exn.what());
    } catch (const std::range_error& exn) {
        PyErr_SetString(PyExc_ArithmeticError, exn.what());
    } catch (const std::underflow_error& exn) {
        PyErr_SetString(PyExc_ArithmeticError, exn.what());
    } catch (const std::exception& exn) {
        PyErr_SetString(PyExc_RuntimeError, exn.what());
    }
}

void pydynd::set_broadcast_exception(PyObject *e)
{
    BroadcastException = e;
}

