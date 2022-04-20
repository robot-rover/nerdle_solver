import string
import numpy as np

from .clues import CLUE_TYPES


def apply_operator(operator, left, right):
    if operator == '*':
        return left * right
    elif operator == '/':
        return left // right
    elif operator == '+':
        return left + right
    elif operator == '-':
        return left - right
    raise RuntimeError(f"Invalid Operator {operator}")


def parse(text):
    chars = text.replace(' ', '')
    tokens = []
    for char in chars:
        if char in string.digits:
            if len(tokens) > 0 and isinstance(tokens[-1], int):
                tokens[-1] = tokens[-1] * 10 + int(char)
            else:
                tokens.append(int(char))
        elif char in '+-/*=':
            tokens.append(char)
        else:
            raise RuntimeError(f'Invalid character in equation: "{char}"')
    return tokens


def evaluate(tokens):
    order = ['*/', '+-']
    for level in order:
        idx = 0
        while idx < len(tokens):
            operator = tokens[idx]
            if isinstance(operator, str) and operator in level:
                if idx < 1 or not isinstance(tokens[idx - 1], int):
                    return None
                left = tokens[idx - 1]
                if idx >= len(tokens) - 1 or not isinstance(tokens[idx + 1], int):
                    return None
                right = tokens[idx + 1]
                tokens = tokens[:idx - 1] + tokens[idx + 1:]
                if operator == '/':
                    if right == 0:
                        return None
                    if (left // right) * right != left:
                        return None
                tokens[idx - 1] = apply_operator(operator, left, right)
            else:
                idx += 1

    if len(tokens) > 1:
        return None
    return tokens[0]


def validate(tokens):
    if len(tokens) < 3:
        return False
    result = tokens[-1]
    if not isinstance(result, int):
        return False
    if tokens[-2] != '=':
        return False

    return evaluate(tokens[:-2]) == result

INDEX_TO_CHAR = np.array(list("0123456789+-*/="))
CHAR_TO_INDEX = {INDEX_TO_CHAR[index]: index for index in range(len(INDEX_TO_CHAR))}
INDEX_TO_CLUE = np.array(list(CLUE_TYPES))

def eq_to_array(equation):
    slots = len(equation)
    array = np.zeros((slots), dtype=np.uint8)
    for x, char in enumerate(equation):
        array[x] = CHAR_TO_INDEX[char]
    return array

def eqs_to_array(equations):
    slots = len(equations[0])
    array = np.zeros((len(equations), slots), dtype=np.uint8)
    for x, eq in enumerate(equations):
        for y, char in enumerate(eq):
            array[x,y] = CHAR_TO_INDEX[char]
    return array

def array_to_eqs(array):
    return [
        ''.join(INDEX_TO_CHAR[array[idx,:]]) for idx in range(array.shape[0])
    ]

def array_to_clues(array):
    return [
        ''.join(INDEX_TO_CLUE[array[idx,:]]) for idx in range(array.shape[0])
    ]