import unittest

import numpy as np

from nerdle_solver.equation import array_to_clues, array_to_eqs, eqs_to_array

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
        clues = np.zeros((len(cases),1,8), dtype=np.uint8)
        with PythonClueContext(len(cases),1) as ctx:
            pass
            print('A')
            ctx.generate_clue(eq_array, guess_arr, clues)
            print('B')
        clues_str = array_to_clues(clues.squeeze())
        print('C')
        for idx, clue in enumerate(clues_str):
            print(clue)
            self.assertEqual(clue, cases[idx][1], cases[idx])
            print('Checked')
        print('Done')

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