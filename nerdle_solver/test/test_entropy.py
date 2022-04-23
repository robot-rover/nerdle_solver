import unittest
from math import log2

import numpy as np
from nerdle_solver.clues import generate_clue, generate_cluev

from nerdle_solver.convert import eq_to_array, eqs_to_array, pack_array

from ..entropy import expected_entropy, expected_entropyv, generate_entropies

class TestEntropy(unittest.TestCase):
    @staticmethod
    def calc_E(counts, num_secret):
        ps = (n/num_secret for n in counts)
        E = sum(p*log2(1/p) for p in ps)
        return E

    def test(self):
        cases = [
            # Guess    N(x)s   Secrets
            ('0123',  [2, 1], ['0122', '0124', '0123']),
            ('1234',  [2], ['0123', '0321']),
        ]
        for case in cases:
            E = TestEntropy.calc_E(case[1], len(case[2]))
            guess = eq_to_array(case[0])
            secrets = eqs_to_array(case[2])
            clues = generate_cluev(secrets, guess)
            clues_packed = pack_array(clues)
            self.assertEqual(expected_entropy(clues_packed, secrets.shape[0]), E, case)

    eqs = ['5+6*5=35', '5*6+5=35', '6*5+5=35', '7+6*5=37', '9+6*5=39']

    def testv(self):
        eqs = eqs_to_array(TestEntropy.eqs)
        clues = np.zeros((eqs.shape[0], eqs.shape[0]), dtype=np.uint16)
        entropies_exp = []
        for guess_idx in range(eqs.shape[0]):
            clues[guess_idx,:] = pack_array(generate_cluev(eqs, eqs[guess_idx]))
            entropies_exp.append(expected_entropy(clues[guess_idx,:], eqs.shape[0]))

        entropies_tup = expected_entropyv(clues, eqs, eqs.shape[0])
        entropies_act = [entropy for eq, entropy in entropies_tup]
        np.testing.assert_equal(entropies_act, entropies_exp, "Testing expectend_entropyv")

    def test_gpu(self):
        eqs = eqs_to_array(TestEntropy.eqs)
        packed_eqs = pack_array(eqs, ord=15, dtype=np.uint64)
        clues = np.zeros((eqs.shape[0], eqs.shape[0]), dtype=np.uint16)
        for guess_idx in range(eqs.shape[0]):
            clues[guess_idx,:] = pack_array(generate_cluev(eqs, eqs[guess_idx]))
        entropies_exp = expected_entropyv(clues, packed_eqs, eqs.shape[0])

        entropies_act = generate_entropies(eqs, eqs, progress=False)
        np.testing.assert_equal(entropies_act, entropies_exp, "Testing expectend_entropyg")
