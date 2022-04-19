#include <Python.h>
#include <cstdio>
#include <iostream>

#define PY_ARRAY_UNIQUE_SYMBOL gen_ARRAY_API
#include "numpy/arrayobject.h"

#include "generate_features.h"

// Module method definitions
static PyObject* generate_features_cpp(PyObject *self, PyObject *args) {
    srand(time(NULL));

    char *filename, *ref, *region;
    PyObject* dict = NULL;

    if (!PyArg_ParseTuple(args, "sssO", &filename, &ref, &region, &dict)) return NULL;
    FeatureGenerator feature_generator {filename, ref, region, dict};
    auto result = feature_generator.generate_features();
    PyObject* return_tuple = PyTuple_New(4);
    PyObject* pos_list = PyList_New(result->positions.size());
    PyObject* X_list = PyList_New(result->X.size());
    PyObject* X2_list = PyList_New(result->X2.size());
    PyObject* Y_list = PyList_New(result->Y.size());

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
        PyList_SetItem(X2_list, i, result->X2[i]);
        PyList_SetItem(Y_list, i, result->Y[i]);
    }
 
    PyTuple_SetItem(return_tuple, 0, pos_list);
    PyTuple_SetItem(return_tuple, 1, X_list);
    PyTuple_SetItem(return_tuple, 2, Y_list);
    PyTuple_SetItem(return_tuple, 3, X2_list);

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
