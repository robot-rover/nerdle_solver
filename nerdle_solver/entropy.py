from math import log2

import numpy as np
from nerdle_solver.clues import generate_cluev
from nerdle_solver.equation import array_to_clues, eqs_to_array

def expected_entropy(guess, possible_secrets):
    clues_slots = generate_cluev(possible_secrets, guess)

    clues_summing_array = 3**(np.arange(len(guess)))
    clue_codes = np.sum(clues_slots * clues_summing_array[None,:], axis=1)
    clue_codes_unique, counts = np.unique(clue_codes, return_counts=True)

    entropy = 0
    num_total = len(possible_secrets)
    probability = counts / num_total
    entropy = probability * np.log2(1/probability)
    return sum(entropy)
