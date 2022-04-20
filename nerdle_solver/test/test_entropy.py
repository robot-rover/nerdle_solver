import unittest
from math import log2
from nerdle_solver.clues import generate_clue, generate_cluev

from nerdle_solver.equation import eq_to_array, eqs_to_array

from ..entropy import expected_entropy

class TestEntropy(unittest.TestCase):
    def test(self):
        cases = [
            # Guess    N(x)s   Secrets
            ('0123',  [2, 1], ['0122', '0124', '0123']),
            ('1234',  [2], ['0123', '0321']),
        ]
        for case in cases:
            ps = (n/len(case[2]) for n in case[1])
            E = sum(p*log2(1/p) for p in ps)
            guess = eq_to_array(case[0])
            secrets = eqs_to_array(case[2])
            clues = generate_cluev(secrets, guess)
            self.assertEqual(expected_entropy(clues, secrets.shape[0]), E, case)