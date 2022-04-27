import math
import os.path
import re
import string

import numpy as np
from nerdle_solver.convert import eqs_to_array, pack_array

from nerdle_solver.equation import parse, evaluate

# Manual Interning
OPERATORS = ('+', '-', '*', '/')
EQUALS = '='


class Counts:
    def __init__(self, slots):
        self.slots = slots
        self.valid = 0
        self.answers = []


def get_all_valid(slots):
    counts = Counts(slots)
    recurse([], counts)
    return counts


def recurse(stack, counts):
    if len(stack) + 1 == counts.slots:
        return
    for digit in string.digits:
        stack.append(digit)
        recurse(stack, counts)
        stack.pop()
    if len(stack) > 0 and stack[-1] in string.digits:
        for operator in OPERATORS:
            stack.append(operator)
            recurse(stack, counts)
            stack.pop()
    # Check Equals
    if len(stack) > 0:
        space_left = counts.slots - (len(stack) + 1)
        answer = evaluate(parse(''.join(stack)))
        if answer is not None and answer >= 0:
            # Log of 0 is undefined but as an int it is displayed with 1 digit
            answer_len = 1 if answer == 0 else math.floor(math.log10(answer) + 1)
            if space_left >= answer_len:
                counts.valid += 1
                text = f"{''.join(stack)}={answer:0{space_left}}"
                counts.answers.append(text)


def _get_comb_list(slots=8):
    cache_path = f'combinations{slots}.txt'
    if not os.path.exists(cache_path):
        print('Generating Combinations...')
        counts = get_all_valid(slots)
        with open(cache_path, 'w') as cache_file:
            cache_file.write('\n'.join(counts.answers))
        return counts.answers
    else:
        with open(cache_path, 'r') as cache_file:
            return cache_file.read().splitlines()

def _get_sol_list(comb_list):
    leading_zeroes = re.compile(r'(?:^|\D)0\d')
    zero_operator = re.compile(r'(?:^|\D)0+(?:$|\D)')
    sol = [ans for ans in comb_list if not leading_zeroes.search(ans) if not zero_operator.search(ans)]
    return sol

def _get_sol_list_indexed(SOL_LIST, COM_LIST):
    indexes = {eq: idx for idx, eq in enumerate(COM_LIST)}
    return [(eq, indexes[eq]) for eq in SOL_LIST]

COM_LIST = _get_comb_list(8)
COM_ARRAY = eqs_to_array(COM_LIST)
COM_PACKED = pack_array(COM_ARRAY, 15, dtype=np.uint64)
SOL_LIST = _get_sol_list(COM_LIST)
SOL_LIST_INDEXED = _get_sol_list_indexed(SOL_LIST, COM_LIST)
SOL_ARRAY = eqs_to_array(SOL_LIST)
SOL_PACKED = pack_array(SOL_ARRAY, 15, dtype=np.uint64)
