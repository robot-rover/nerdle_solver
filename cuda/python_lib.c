#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>

#define NUM_SLOTS 8

extern double * myVectorAdd(double * h_A, double * h_B, int numElements);
extern uint8_t *generate_clueg(uint8_t *secrets, uint8_t *guess, uint64_t num_batches);

static PyObject* helloworld(PyObject* self, PyObject* args) {
    // printf("Hello World\n");
    return Py_None;
}

static PyObject* vector_add_cuda(PyObject* self, PyObject* args) {
    PyArrayObject* array1, * array2;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &array1, &PyArray_Type, &array2))
        return NULL;

    if (array1 -> nd != 1 || array2 -> nd != 1 || array1->descr->type_num != PyArray_DOUBLE || array2->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "arrays must be one-dimensional and of type float");
        return NULL;
    }

    int n1 = array1->dimensions[0];
    int n2 = array2->dimensions[0];

    // printf("running vector_add on dim1: %d, stride1: %d, dim2: %d, stride2: %d\n", n1, array1->strides[0], n2, array2->strides[0]);

    if (n1 != n2) {
        PyErr_SetString(PyExc_ValueError, "arrays must have the same length");
        return NULL;
    }

    double * output = myVectorAdd((double *)array1->data, (double *)array2->data, n1);

    return PyArray_SimpleNewFromData(1, PyArray_DIMS(array1), PyArray_TYPE(array1), output);
}

static PyObject* vector_add(PyObject* self, PyObject* args) {
    PyArrayObject* array1, * array2;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &array1, &PyArray_Type, &array2))
        return NULL;

    if (array1 -> nd != 1 || array2 -> nd != 1 || array1->descr->type_num != PyArray_DOUBLE || array2->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "arrays must be one-dimensional and of type float");
        return NULL;
    }

    int n1 = array1->dimensions[0];
    int n2 = array2->dimensions[0];

    // printf("running vector_add on dim1: %d, stride1: %d, dim2: %d, stride2: %d\n", n1, array1->strides[0], n2, array2->strides[0]);

    if (n1 != n2) {
        PyErr_SetString(PyExc_ValueError, "arrays must have the same length");
        return NULL;
    }

    double * output = (double *) malloc(sizeof(double) * n1);

    for (int i = 0; i < n1; i++)
        output[i] = *((double *) array1 -> data + i) + *((double *) array2 -> data + i);

    return PyArray_SimpleNewFromData(1, PyArray_DIMS(array1), PyArray_TYPE(array1), output);
}

static PyObject* generate_clue_gpu(PyObject* self, PyObject* args) {
    PyArrayObject *secrets, *guess;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &secrets, &PyArray_Type, &guess))
        return NULL; // Python throws the correct error if ParseTuple fails

    if (secrets -> nd != 2 || secrets->descr->type_num != NPY_UINT8) {
        PyErr_SetString(PyExc_ValueError, "secrets must be 2 dimensional and of type uint8");
        return NULL;
    }

    if (guess -> nd != 1 || guess->descr->type_num != NPY_UINT8) {
        PyErr_SetString(PyExc_ValueError, "guess must be 1 dimensional and of type uint8");
        return NULL;
    }

    size_t guess_width = guess->dimensions[0];
    size_t num_batches = secrets->dimensions[0];
    size_t secrets_width = secrets->dimensions[1];

    if (guess_width != NUM_SLOTS || secrets_width != NUM_SLOTS) {
        PyErr_SetString(PyExc_ValueError, "arrays must have a width of NUM_SLOTS (8)");
        return NULL;
    }

    uint8_t *clues = generate_clueg(secrets->data, guess->data, num_batches);

    return PyArray_SimpleNewFromData(2, PyArray_DIMS(secrets), NPY_UINT8, clues);
}

static PyMethodDef methods[] = {
    {"helloworld", helloworld, METH_NOARGS, "A Simple Hello World Function"}, // (function name, function, arguments, doc_string)
    {"vector_add", vector_add, METH_VARARGS, "add two numpy float arrays together on the CPU"},
    {"vector_add_cuda", vector_add_cuda, METH_VARARGS, "add two numpy float arrays together on the GPU"},
    {"generate_clue_gpu", generate_clue_gpu, METH_VARARGS, "nerdle clue GPU"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef nerdle_cuda = {
    PyModuleDef_HEAD_INIT, "nerdle_cuda_ext", // name of the module
    "nerdle_cuda_ext", -1, methods
};

PyMODINIT_FUNC PyInit_nerdle_cuda_ext(void) {
    import_array();
    return PyModule_Create(&nerdle_cuda);
}
