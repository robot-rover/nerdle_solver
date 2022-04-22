import math
import os.path
import re
import string

import numpy as np
from nerdle_solver.convert import eqs_to_array

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


def get_comb_list(slots=8):
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

def get_comb_array(slots=8):
    cache_path = f'combinations{slots}.npy'
    if not os.path.exists(cache_path):
        comb_list = get_comb_list(slots)
        comb_array = eqs_to_array(comb_list)
        np.save(cache_path, comb_array, allow_pickle=False)
        return comb_array
    else:
        return np.load(cache_path, allow_pickle=False)

def get_sol_array(slots=8):
    cache_path = f'solutions{slots}.npy'
    if not os.path.exists(cache_path):
        sols_list = get_sol_list(slots)
        sols_array = eqs_to_array(sols_list)
        np.save(cache_path, sols_array, allow_pickle=False)
        return sols_array
    else:
        return np.load(cache_path, allow_pickle=False)

def get_sol_list(slots=8):
    comb = get_comb_list(slots)
    leading_zeroes = re.compile(r'(?:^|\D)0\d')
    zero_operator = re.compile(r'(?:^|\D)0+(?:$|\D)')
    sol = [ans for ans in comb if not leading_zeroes.search(ans) if not zero_operator.search(ans)]
    return sol
