#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>

#define NUM_SLOTS 8

extern double * myVectorAdd(double * h_A, double * h_B, int numElements);
void generate_clueg(uint8_t *guess_eqs, uint32_t num_guess, uint8_t *secret_eqs, uint32_t num_secret, uint8_t *clue_arr);

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
    PyArrayObject *secrets, *guesses, *clues;

    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &secrets, &PyArray_Type, &guesses, &PyArray_Type, &clues))
        return NULL; // Python throws the correct error if ParseTuple fails

    if (secrets -> nd != 2 || secrets->descr->type_num != NPY_UINT8) {
        PyErr_SetString(PyExc_ValueError, "secrets must be 2 dimensional and of type uint8");
        return NULL;
    }

    if (guesses -> nd != 2 || guesses->descr->type_num != NPY_UINT8) {
        PyErr_SetString(PyExc_ValueError, "guesses must be 2 dimensional and of type uint8");
        return NULL;
    }

    if (clues -> nd != 3 || clues->descr->type_num != NPY_UINT8) {
        PyErr_SetString(PyExc_ValueError, "clues must be 3 dimensional and of type uint8");
        return NULL;
    }

    size_t num_secrets = secrets->dimensions[0];
    size_t num_guesses = guesses->dimensions[0];
    size_t num_slots_s = secrets->dimensions[1];
    size_t num_slots_g = guesses->dimensions[1];
    size_t num_slots_c = clues->dimensions[2];
    if (num_slots_s != NUM_SLOTS || num_slots_g != NUM_SLOTS || num_slots_c != NUM_SLOTS) {
        PyErr_SetString(PyExc_ValueError, "arrays last dimension must be NUM_SLOTS (8)");
        return NULL;
    }

    if (num_secrets > clues->dimensions[0] || num_guesses > clues->dimensions[1]) {
        PyErr_SetString(PyExc_ValueError, "clues array isn't big enough");
        return NULL;
    }



    generate_clueg(secrets->data, num_secrets, guesses->data, num_guesses, clues->data);

    return Py_None;
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
