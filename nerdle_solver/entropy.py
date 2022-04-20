from math import log2

import numpy as np
from nerdle_solver.clues import generate_cluev
from nerdle_solver.equation import array_to_clues, eqs_to_array

def expected_entropy(clues_slots, num_secrets):
    clues_summing_array = 3**(np.arange(clues_slots.shape[1]))
    clue_codes = np.sum(clues_slots * clues_summing_array[None,:], axis=1)
    clue_codes_unique, counts = np.unique(clue_codes, return_counts=True)
    _, stats = np.unique(counts, return_counts=True)
    print(f"Expected Entropy stats: {stats}")

    entropy = 0
    probability = counts / num_secrets
    entropy = probability * np.log2(1/probability)
    return sum(entropy)
