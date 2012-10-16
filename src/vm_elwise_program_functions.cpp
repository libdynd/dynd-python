//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "vm_elwise_program_functions.hpp"
#include "utility_functions.hpp"
#include "dtype_functions.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

void pydynd::vm_elwise_program_from_py(PyObject *obj, dynd::vm::elwise_program& out_ep)
{
    vector<dtype> regtypes;
    vector<int> program;
    int input_count;

    if (!PyDict_Check(obj)) {
        throw runtime_error("Expected a Python dict to convert to a VM program");
    }

    // The number of inputs

    input_count = pyobject_as_int_index(pydict_getitemstring(obj, "num_inputs"));

    // The list of register types
    PyObject *regtypes_object = pydict_getitemstring(obj, "register_types");
    Py_ssize_t regtypes_size = pysequence_size(regtypes_object);
    regtypes.resize(regtypes_size);
    for (Py_ssize_t i = 0; i < regtypes_size; ++i) {
        pyobject_ownref item(PySequence_GetItem(regtypes_object, i));
        regtypes[i] = make_dtype_from_object(item);
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
        char *opcode_str;
        Py_ssize_t opcode_size;
        int opcode = -1;
        if (PyString_AsStringAndSize(PyTuple_GET_ITEM(instr.get(), 0), &opcode_str, &opcode_size) < 0)
            throw exception();
        switch (opcode_size) {
            case 3:
                if (!strcmp(opcode_str, "add"))
                    opcode = vm::opcode_add;
                break;
            case 4:
                if (!strcmp(opcode_str, "copy"))
                    opcode = vm::opcode_copy;
                break;
            case 7:
                if (!strcmp(opcode_str, "divide"))
                    opcode = vm::opcode_divide;
                break;
            case 8:
                if (!strcmp(opcode_str, "subtract"))
                    opcode = vm::opcode_subtract;
                else if (!strcmp(opcode_str, "multiply"))
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
