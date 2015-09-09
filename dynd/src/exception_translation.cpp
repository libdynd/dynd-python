//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include <stdexcept>
#include <sstream>

#include <dynd/exceptions.hpp>

#include "exception_translation.hpp"

namespace {

PyObject *BroadcastException = NULL;
} // anonymous namespace

void pydynd::translate_exception()
{
  try {
    // let any Python exn pass through, otherwise translate the C++ one
    if (!PyErr_Occurred()) {
      throw;
    }
  }
  catch (const dynd::broadcast_error &exn) {
    PyErr_SetString(BroadcastException, exn.message());
  }
  catch (const dynd::too_many_indices &exn) {
    PyErr_SetString(PyExc_IndexError, exn.message());
  }
  catch (const dynd::index_out_of_bounds &exn) {
    PyErr_SetString(PyExc_IndexError, exn.message());
  }
  catch (const dynd::axis_out_of_bounds &exn) {
    PyErr_SetString(PyExc_IndexError, exn.message());
  }
  catch (const dynd::irange_out_of_bounds &exn) {
    PyErr_SetString(PyExc_IndexError, exn.message());
  }
  catch (const dynd::invalid_type_id &exn) {
    PyErr_SetString(PyExc_TypeError, exn.message());
  }
  catch (const dynd::type_error &exn) {
    PyErr_SetString(PyExc_TypeError, exn.message());
  }
  catch (const dynd::string_encode_error &exn) {
    std::stringstream ss;
    ss << exn.encoding();
    // TODO: In Python 3.3, Py_UNICODE might be 16-bit, but
    //       the string still supports 32-bit, should fix
    //       this up to generate an appropriate unicode
    //       string in that case.
    Py_UNICODE dummy[1] = {(Py_UNICODE)exn.cp()};
    PyErr_SetObject(PyExc_UnicodeEncodeError,
                    Py_BuildValue("su#nns", ss.str().c_str(), &dummy[0], 1, 0,
                                  1, exn.message()));
  }
  catch (const dynd::string_decode_error &exn) {
    std::stringstream ss;
    ss << exn.encoding();
    const std::string &bytes = exn.bytes();
#if PY_VERSION_HEX >= 0x03000000
    PyErr_SetObject(PyExc_UnicodeDecodeError,
                    Py_BuildValue("sy#nns", ss.str().c_str(), bytes.data(),
                                  bytes.size(), 0, (int)bytes.size(),
                                  exn.message()));
#else
    PyErr_SetObject(PyExc_UnicodeDecodeError,
                    Py_BuildValue("ss#nns", ss.str().c_str(), bytes.data(),
                                  bytes.size(), 0, (int)bytes.size(),
                                  exn.message()));
#endif
  }
  catch (const std::bad_alloc &exn) {
    PyErr_SetString(PyExc_MemoryError, exn.what());
    //    } catch (const std::bad_cast& exn) {
    //        PyErr_SetString(PyExc_TypeError, exn.what());
  }
  catch (const std::domain_error &exn) {
    PyErr_SetString(PyExc_ValueError, exn.what());
  }
  catch (const std::invalid_argument &exn) {
    PyErr_SetString(PyExc_ValueError, exn.what());
  }
  catch (const std::out_of_range &exn) {
    PyErr_SetString(PyExc_IndexError, exn.what());
  }
  catch (const std::overflow_error &exn) {
    PyErr_SetString(PyExc_OverflowError, exn.what());
  }
  catch (const std::range_error &exn) {
    PyErr_SetString(PyExc_ArithmeticError, exn.what());
  }
  catch (const std::underflow_error &exn) {
    PyErr_SetString(PyExc_ArithmeticError, exn.what());
  }
  catch (const std::runtime_error &exn) {
    // In some circumstances (built on OSX 10.7,
    // run on OSX 10.8), this exception wasn't being
    // caught by the "std::exception" catch, so
    // redundantly also catch runtime_error.
    PyErr_SetString(PyExc_RuntimeError, exn.what());
  }
  catch (const dynd::dynd_exception &exn) {
    PyErr_SetString(PyExc_RuntimeError, exn.what());
  }
  catch (const std::exception &exn) {
    PyErr_SetString(PyExc_RuntimeError, exn.what());
  }
}

void pydynd::set_broadcast_exception(PyObject *e) { BroadcastException = e; }
