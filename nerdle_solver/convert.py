import numpy as np

from nerdle_solver.clues import CLUE_TYPES

INDEX_TO_CHAR = np.array(list("0123456789+-*/="))
CHAR_TO_INDEX = {INDEX_TO_CHAR[index]: index for index in range(len(INDEX_TO_CHAR))}
INDEX_TO_CLUE = np.array(list(CLUE_TYPES))

def eq_to_array(equation):
    slots = len(equation)
    array = np.zeros((slots), dtype=np.uint8)
    for x, char in enumerate(equation):
        array[x] = CHAR_TO_INDEX[char]
    return array

def eqs_to_array(equations, slots=None, quantity=None):
    if slots is None:
        slots = len(equations[0])
    if quantity is None:
        quantity = len(equations)
    array = np.zeros((quantity, slots), dtype=np.uint8)
    for x, eq in enumerate(equations):
        for y, char in enumerate(eq):
            array[x,y] = CHAR_TO_INDEX[char]
    return array

def array_to_eqs(array):
    return [
        ''.join(INDEX_TO_CHAR[array[idx,:]]) for idx in range(array.shape[0])
    ]

def array_to_eq(array):
    return ''.join(INDEX_TO_CHAR[array])

def array_to_clues(array):
    return [
        ''.join(INDEX_TO_CLUE[array[idx,:]]) for idx in range(array.shape[0])
    ]

def unpack_array(array, slots=8, ord=3):
    unpacked = (array[...,np.newaxis] // ord**np.arange(slots, dtype=array.dtype)[::-1]) % ord
    return unpacked

def pack_array(array, ord=3, dtype=np.uint16):
    packed = np.sum(array.astype(dtype) * ord**np.arange(array.shape[-1], dtype=dtype)[::-1], axis=-1)
    return packed

def packed_array_to_clues(array):
    unpacked = unpack_array(array)
    return array_to_clues(unpacked)