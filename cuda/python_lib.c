#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <numpy/arrayobject.h>
#include "cuda_lib.h"

#define NUM_SLOTS 8

static PyObject* helloworld(PyObject* self, PyObject* args) {
    printf("Hello World\n");
    return Py_None;
}

typedef struct {
    PyObject_HEAD
    size_t num_secret;
    size_t num_guess;
    ClueContext* ctx;
} PythonClueContext;

static int PythonClueContext_init(PythonClueContext *self, PyObject *args, PyObject *kwds)
{
    Py_ssize_t num_secret, num_guess;
    if (!PyArg_ParseTuple(args, "nn", &num_secret, &num_guess))
        return -1;
    self->ctx = NULL;
    self->num_secret = num_secret;
    self->num_guess = num_guess;
    return 0;
}

static void PythonClueContext_dealloc(PythonClueContext *self)
{
    if (self->ctx != NULL) {
        free_context(self->ctx);
        self->ctx = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*) self);
}

static PyObject* PythonClueContext_enter(PythonClueContext *self, PyObject *Py_UNUSED(ignored))
{
    if (self->ctx != NULL)
    {
        PyErr_SetString(PyExc_ValueError, "PythonClueContext is already opened");
        return NULL;
    }
    self->ctx = create_context(self->num_secret, self->num_guess);
    return (PyObject*) self;
}

static PyObject* PythonClueContext_exit(PythonClueContext *self, PyObject *Py_UNUSED(ignored))
{
    if (self->ctx == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "PythonClueContext isn't opened yet");
        return NULL;
    }
    free_context(self->ctx);
    self->ctx = NULL;
    return Py_None;
}

static PyObject* PythonClueContext_generate_clue(PythonClueContext *self, PyObject *args) {
    PyArrayObject *secrets, *guesses, *clues;

    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &secrets, &PyArray_Type, &guesses, &PyArray_Type, &clues))
        return NULL;

    if (self->ctx == NULL) {
        PyErr_SetString(PyExc_ValueError, "context is not opened");
        return NULL;
    }

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

    if (num_secrets > (size_t)clues->dimensions[0] || num_guesses > (size_t)clues->dimensions[1]) {
        PyErr_SetString(PyExc_ValueError, "clues array isn't big enough");
        return NULL;
    }

    int result = generate_clueg(self->ctx, secrets->data, num_secrets, guesses->data, num_guesses, clues->data);
    if (result < 0) {
        PyErr_Format(PyExc_IOError, "CUDA error (%d)", result);
        return NULL;
    }

    return Py_None;
}

static PyMethodDef PythonClueContext_methods[] = {
    {"__enter__", (PyCFunction) PythonClueContext_enter, METH_NOARGS,
    "Initialize Context"},
    {"__exit__", (PyCFunction) PythonClueContext_exit, METH_VARARGS,
    "Free Context"},
    {"generate_clue", (PyCFunction) PythonClueContext_generate_clue, METH_VARARGS,
    "Generate Clue"},
    {NULL}  /* Sentinel */
};

static PyTypeObject PythonClueContextType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "nerdle_cuda_ext.PythonClueContext",
    .tp_doc = "Python Cuda Clue Context",
    .tp_basicsize = sizeof(PythonClueContext),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) PythonClueContext_init,
    .tp_dealloc = (destructor) PythonClueContext_dealloc,
    .tp_methods = PythonClueContext_methods,
};

static PyMethodDef methods[] = {
    {"helloworld", helloworld, METH_NOARGS, "A Simple Hello World Function"}, // (function name, function, arguments, doc_string)
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef nerdle_cuda = {
    PyModuleDef_HEAD_INIT,
    .m_name = "nerdle_cuda_ext",
    .m_doc = "Cuda Nerdle Extension",
    .m_size = -1,
    .m_methods = methods,
};

PyMODINIT_FUNC PyInit_nerdle_cuda_ext(void) {
    import_array();
    PyObject *m;
    if (PyType_Ready(&PythonClueContextType) < 0) {
        return NULL;
    }

    m = PyModule_Create(&nerdle_cuda);
    if (m == NULL)
        return NULL;

    Py_INCREF(&PythonClueContextType);
    if (PyModule_AddObject(m, "PythonClueContext", (PyObject*) &PythonClueContextType) < 0) {
        Py_DECREF(&PythonClueContextType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
