import unittest

import numpy as np
import sys

from nerdle_solver.convert import array_to_clues, array_to_eqs, eq_to_array, eqs_to_array, pack_array, packed_array_to_clues, unpack_array

from ..clues import generate_clue, generate_cluev
from nerdle_cuda import PythonClueContext

class TestClue(unittest.TestCase):
    def test(self):
        cases = [
            # Secret      Guess       Clue
            ('10+20=30', '10+10=20', 'gggbggyg'),
            ('10+10=20', '20+21=41', 'yggbygby'),
            ('27+20=47', '000001=1', 'bbbbgbyb'),
            ('27+20=47', '9/3-00=3', 'bbbbgbyb'),
            ('27+20=47', '24+54=78', 'gygbbgyb'),
            ('27+20=47', '27+64=91', 'gggbygbb'),
            ('27+20=47', '27+47=74', 'gggyygbb'),
            ('27+20=47', '27+20=47', 'gggggggg'),
            (   'weary',    'weays',    'gggyb'),
        ]
        for case in cases:
            self.assertEqual(generate_clue(case[0], case[1]), case[2])

    def test_cluev(self):
        guess = '27+20=47'
        cases = [
            #Secret       Clue
            ('000001=1', 'bbbbgybb'),
            ('9/3-00=3', 'bbbbgybb'),
            ('24+54=78', 'gygbbgyb'),
            ('27+64=91', 'gggbbgyb'),
            ('27+47=74', 'gggbbgyy'),
            ('27+20=47', 'gggggggg'),
        ]
        eq_array = eqs_to_array([case[0] for case in cases])
        guess_arr = eqs_to_array([guess]).squeeze()
        clues = generate_cluev(eq_array, guess_arr)
        clues_str = array_to_clues(clues)
        for idx, clue in enumerate(clues_str):
            self.assertEqual(clue, cases[idx][1], cases[idx])

    def test_clueg(self):
        guess = '27+20=47'
        cases = [
            #Secret       Clue
            ('000001=1', 'bbbbgybb'),
            ('9/3-00=3', 'bbbbgybb'),
            ('24+54=78', 'gygbbgyb'),
            ('27+64=91', 'gggbbgyb'),
            ('27+47=74', 'gggbbgyy'),
            ('27+20=47', 'gggggggg'),
        ]
        eq_array = eqs_to_array([case[0] for case in cases])
        guess_arr = eqs_to_array([guess])
        clues = np.zeros((1, len(cases)), dtype=np.uint16)
        with PythonClueContext(1, len(cases)) as ctx:
            ctx.generate_clue(guess_arr, eq_array, clues)
        clues_str = packed_array_to_clues(clues.squeeze())
        for idx, clue in enumerate(clues_str):
            self.assertEqual(clue, cases[idx][1], f'Case {idx}')

    def test_compare(self):
        eqs = [
            "000001=1",
            "9/3+2=05",
            "9/3+2=05",
            "9/3+2=05",
            '24+54=78',
            '27+64=91',
            '27+47=74',
            '27+20=47'
        ]

        python_clues = [
            [generate_clue(secret, guess) for secret in eqs] for guess in eqs
        ]

        eqs_array = eqs_to_array(eqs)
        numpy_clues = [
            array_to_clues(generate_cluev(eqs_array, eq_to_array(guess))) for guess in eqs
        ]

        gpu_clues_packed = np.zeros((len(eqs), len(eqs)), dtype=np.uint16)
        with PythonClueContext(len(eqs), len(eqs)) as ctx:
           ctx.generate_clue(eqs_array, eqs_array, gpu_clues_packed)

        gpu_clues = [
            array_to_clues(unpack_array(gpu_clues_row)) for gpu_clues_row in gpu_clues_packed
        ]

        for guess, pyclue, npclue in zip(eqs, python_clues, numpy_clues):
            self.assertSequenceEqual(pyclue, npclue, f'PvN Guess {guess}')
        for guess, npclue, gpuclue in zip(eqs, numpy_clues, gpu_clues):
            self.assertSequenceEqual(npclue, gpuclue, f'NvG Guess {guess}')

    def test_eq_to_array(self):
        cases = [
            (['0123456789+-*/='], [np.arange(15)]),
            (['000000=0', '15+15=30'], np.stack([np.array([0]*6 + [14, 0]), np.array([1,5,10,1,5,14,3,0])]))
        ]
        for case in cases:
            arr = eqs_to_array(case[0])
            for idx, eq in enumerate(arr):
                np.testing.assert_array_equal(eq, case[1][idx], f'Case: {case}, Subeq: {idx}')

    def test_array_to_eq(self):
        cases = [
            (['0123456789+-*/='], np.arange(15)[None,:]),
            (['000000=0', '15+15=30'], np.stack([np.array([0]*6 + [14, 0]), np.array([1,5,10,1,5,14,3,0])]))
        ]
        for case in cases:
            eqs = array_to_eqs(case[1])
            for idx, eqex in enumerate(eqs):
                self.assertEqual(case[0][idx], eqex, f'Case: {case}, Subeq: {idx}')