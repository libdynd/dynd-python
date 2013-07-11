//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "vm_elwise_program_functions.hpp"
#include "utility_functions.hpp"
#include "type_functions.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

void pydynd::vm_elwise_program_from_py(PyObject *obj, dynd::vm::elwise_program& out_ep)
{
    vector<ndt::type> regtypes;
    vector<int> program;
    int input_count;

    if (!PyDict_Check(obj)) {
        throw runtime_error("Expected a Python dict to convert to a VM program");
    }

    // The number of inputs

    input_count = pyobject_as_int_index(pydict_getitemstring(obj, "input_count"));

    // The list of register types
    PyObject *regtypes_object = pydict_getitemstring(obj, "register_types");
    Py_ssize_t regtypes_size = pysequence_size(regtypes_object);
    regtypes.resize(regtypes_size);
    for (Py_ssize_t i = 0; i < regtypes_size; ++i) {
        pyobject_ownref item(PySequence_GetItem(regtypes_object, i));
        regtypes[i] = make_ndt_type_from_pyobject(item.get());
    }

    // The program (list of instructions)
    PyObject *program_object = pydict_getitemstring(obj, "program");
    Py_ssize_t program_size = pysequence_size(program_object);
    for (Py_ssize_t i = 0; i < program_size; ++i) {
        pyobject_ownref instr(PySequence_GetItem(program_object, i));
        if (!PyTuple_Check(instr)) {
            throw runtime_error("Each instruction in the VM program must be a tuple");
        }
        Py_ssize_t instr_size = PyTuple_GET_SIZE(instr.get());
        if (instr_size == 0) {
            throw runtime_error("Each instruction in the VM program must have at least an opcode");
        }
        // The opcode
        string opcode_str = pystring_as_string(PyTuple_GET_ITEM(instr.get(), 0));
        int opcode = -1;
        switch (opcode_str.size()) {
            case 3:
                if (opcode_str == "add")
                    opcode = vm::opcode_add;
                break;
            case 4:
                if (opcode_str == "copy")
                    opcode = vm::opcode_copy;
                break;
            case 7:
                if (opcode_str == "divide")
                    opcode = vm::opcode_divide;
                break;
            case 8:
                if (opcode_str == "subtract")
                    opcode = vm::opcode_subtract;
                else if (opcode_str == "multiply")
                    opcode = vm::opcode_multiply;
                break;
        }
        if (opcode == -1) {
            stringstream ss;
            ss << "Unrecognized opcode '" << opcode_str << "'";
            throw runtime_error(ss.str());
        }
        program.push_back(opcode);
        // The registers (this is validated later, just copy them here)
        for (Py_ssize_t i = 1; i < instr_size; ++i) {
            int reg = pyobject_as_int_index(PyTuple_GET_ITEM(instr.get(), i));
            program.push_back(reg);
        }
    }

    out_ep.set(input_count, regtypes, program);
}

PyObject *pydynd::vm_elwise_program_as_py(dynd::vm::elwise_program& ep)
{
    pyobject_ownref regtypes_obj(PyList_New(ep.get_register_types().size()));
    pyobject_ownref program_obj(PyList_New(ep.get_instruction_count()));
    pyobject_ownref input_count(PyLong_FromLong(ep.get_input_count()));

    // Set the list of register types
    for (size_t i = 0; i < ep.get_register_types().size(); ++i) {
        PyList_SET_ITEM(regtypes_obj.get(), i, wrap_ndt_type(ep.get_register_types()[i]));
    }

    // Set the list of instructions
    int ip = 0;
    for (int i = 0; i < ep.get_instruction_count(); ++i) {
        int opcode = ep.get_program()[ip];
        int arity = vm::opcode_info[opcode].arity;
        pyobject_ownref instr(PyTuple_New(arity + 2));
#if PY_VERSION_HEX >= 0x03000000
        pyobject_ownref name_str(PyUnicode_FromString(vm::opcode_info[opcode].name));
#else
        pyobject_ownref name_str(PyString_FromString(vm::opcode_info[opcode].name));
#endif
        PyTuple_SET_ITEM(instr.get(), 0, name_str.release());
        for (int j = 1; j < 2 + arity; ++j) {
            PyTuple_SET_ITEM(instr.get(), j, PyLong_FromLong(ep.get_program()[ip + j]));
        }
        PyList_SET_ITEM(program_obj.get(), i, instr.release());
        ip += 2 + arity;
    }

    // Pack the values into a dict
    pyobject_ownref result_obj(PyDict_New());
    PyDict_SetItemString(result_obj.get(), "input_count", input_count.get());
    PyDict_SetItemString(result_obj.get(), "register_types", regtypes_obj.get());
    PyDict_SetItemString(result_obj.get(), "program", program_obj.get());

    return result_obj.release();
}

