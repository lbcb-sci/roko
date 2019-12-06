#include <Python.h>
#include <cstdio>

#define PY_ARRAY_UNIQUE_SYMBOL gen_ARRAY_API
#include "numpy/arrayobject.h"

#include "generate.h"

// Module method definitions
static PyObject* generate_features_cpp(PyObject *self, PyObject *args) {
    srand(time(NULL));

    char *filename, *ref, *region;

    if (!PyArg_ParseTuple(args, "sss", &filename, &ref, &region)) return NULL;

    auto result = generate_features(filename, ref, region);

    PyObject* return_tuple = PyTuple_New(2);
    PyObject* pos_list = PyList_New(result->positions.size());
    PyObject* X_list = PyList_New(result->X.size());

    for (int i = 0, size=result->positions.size(); i < size; i++) {
        auto& pos_element = result->positions[i];

        PyObject* inner_list = PyList_New(pos_element.size());
        for (int j = 0, s = pos_element.size(); j < s; j++) {
            PyObject* pos_tuple = PyTuple_New(2);
            PyTuple_SetItem(pos_tuple, 0, PyLong_FromLong(pos_element[j].first));
            PyTuple_SetItem(pos_tuple, 1, PyLong_FromLong(pos_element[j].second));

            PyList_SetItem(inner_list, j, pos_tuple);
        }
        PyList_SetItem(pos_list, i, inner_list);

        PyList_SetItem(X_list, i, result->X[i]);
    }
 
    PyTuple_SetItem(return_tuple, 0, pos_list);
    PyTuple_SetItem(return_tuple, 1, X_list);

    return return_tuple;
}

static PyMethodDef gen_methods[] = {
        {
                "generate_features", generate_features_cpp, METH_VARARGS,
                "Generate features for polisher."
        },
        {NULL, NULL, 0, NULL}
};


static struct PyModuleDef gen_definition = {
        PyModuleDef_HEAD_INIT,
        "gen",
        "Feature generation.",
        -1,
        gen_methods
};


PyMODINIT_FUNC PyInit_gen(void) {
    Py_Initialize();
    import_array();
    return PyModule_Create(&gen_definition);
}
