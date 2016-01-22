//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "array_functions.hpp"
#include "array_from_py.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"
#include "types/pyobject_type.hpp"

#include <dynd/func/assignment.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/base_dim_type.hpp>
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/array_range.hpp>
#include <dynd/types/base_bytes_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/view.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

dynd::nd::array pydynd::array_linspace(PyObject *start, PyObject *stop,
                                       PyObject *count, PyObject *dt)
{
  nd::array start_nd, stop_nd;
  intptr_t count_val = pyobject_as_index(count);
  start_nd = array_from_py(start, 0, false);
  stop_nd = array_from_py(stop, 0, false);
  if (dt == Py_None) {
    return nd::linspace(start_nd, stop_nd, count_val);
  }
  else {
    return nd::linspace(start_nd, stop_nd, count_val,
                        make__type_from_pyobject(dt));
  }
}

dynd::nd::array pydynd::nd_fields(const nd::array &n, PyObject *field_list)
{
  vector<std::string> selected_fields;
  pyobject_as_vector_string(field_list, selected_fields);

  // TODO: Move this implementation into dynd
  ndt::type fdt = n.get_dtype();
  if (fdt.get_kind() != struct_kind) {
    stringstream ss;
    ss << "nd.fields must be given a dynd array of 'struct' kind, not ";
    ss << fdt;
    throw runtime_error(ss.str());
  }
  const ndt::struct_type *bsd = fdt.extended<ndt::struct_type>();

  if (selected_fields.empty()) {
    throw runtime_error(
        "nd.fields requires at least one field name to be specified");
  }
  // Construct the field mapping and output field types
  vector<intptr_t> selected_index(selected_fields.size());
  vector<ndt::type> selected__types(selected_fields.size());
  for (size_t i = 0; i != selected_fields.size(); ++i) {
    selected_index[i] = bsd->get_field_index(selected_fields[i]);
    if (selected_index[i] < 0) {
      stringstream ss;
      ss << "field name ";
      print_escaped_utf8_string(ss, selected_fields[i]);
      ss << " does not exist in dynd type " << fdt;
      throw runtime_error(ss.str());
    }
    selected__types[i] = bsd->get_field_type(selected_index[i]);
  }
  // Create the result udt
  ndt::type rudt = ndt::struct_type::make(selected_fields, selected__types);
  ndt::type result_tp = n.get_type().with_replaced_dtype(rudt);
  const ndt::struct_type *rudt_bsd = rudt.extended<ndt::struct_type>();

  // Allocate the new memory block.
  size_t arrmeta_size = result_tp.get_arrmeta_size();
  nd::array result(reinterpret_cast<array_preamble *>(
                       make_array_memory_block(arrmeta_size).get()),
                   true);

  // Clone the data pointer
  result.get()->data = n.get()->data;
  result.get()->owner = n.get()->owner;
  if (!result.get()->owner) {
    result.get()->owner = n.get();
  }

  // Copy the flags
  result.get()->flags = n.get()->flags;

  // Set the type and transform the arrmeta
  result.get()->tp = result_tp;
  // First copy all the array data type arrmeta
  ndt::type tmp_dt = result_tp;
  char *dst_arrmeta = result.get()->metadata();
  const char *src_arrmeta = n.get()->metadata();
  while (tmp_dt.get_ndim() > 0) {
    if (tmp_dt.get_kind() != dim_kind) {
      throw runtime_error(
          "nd.fields doesn't support dimensions with pointers yet");
    }
    const ndt::base_dim_type *budd = tmp_dt.extended<ndt::base_dim_type>();
    size_t offset = budd->arrmeta_copy_construct_onedim(
        dst_arrmeta, src_arrmeta,
        intrusive_ptr<memory_block_data>(n.get(), true));
    dst_arrmeta += offset;
    src_arrmeta += offset;
    tmp_dt = budd->get_element_type();
  }
  // Then create the arrmeta for the new struct
  const size_t *arrmeta_offsets = bsd->get_arrmeta_offsets_raw();
  const size_t *result_arrmeta_offsets = rudt_bsd->get_arrmeta_offsets_raw();
  const size_t *data_offsets = bsd->get_data_offsets(src_arrmeta);
  size_t *result_data_offsets = reinterpret_cast<size_t *>(dst_arrmeta);
  for (size_t i = 0; i != selected_fields.size(); ++i) {
    const ndt::type &dt = selected__types[i];
    // Copy the data offset
    result_data_offsets[i] = data_offsets[selected_index[i]];
    // Copy the arrmeta for this field
    if (dt.get_arrmeta_size() > 0) {
      dt.extended()->arrmeta_copy_construct(
          dst_arrmeta + result_arrmeta_offsets[i],
          src_arrmeta + arrmeta_offsets[selected_index[i]],
          intrusive_ptr<memory_block_data>(n.get(), true));
    }
  }

  return result;
}

PYDYND_API dynd::nd::array pydynd::pyobject_array(PyObject *obj)
{
  dynd::nd::array a = dynd::nd::empty(dynd::ndt::make_type<pyobject_type>());
  *reinterpret_cast<PyObject **>(a.data()) = obj;

  return a;
}
