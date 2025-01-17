import unittest

from nerdle_solver.combinations import COM_LIST

from ..clues import filter_secrets



class TestFilter(unittest.TestCase):
    def test(self):
        possible_secrets = COM_LIST
        solution1 = ['64/01=64', '64/02=32', '64/04=16', '64/08=08', '64/16=04', '64/32=02', '64/64=01']
        solution2 = [
            '12+30=42', '12+31=43', '12+32=44', '12+33=45', '12+34=46',
            '12+35=47', '12+36=48', '12+37=49', '12+38=50', '12+39=51'
        ]

        cases = [
            # Guess       Clue        Solution
            ('64/XX=XX', 'gggbbgbb', solution1),
            ('12+3X=XX', 'ggggbgbb', solution2),
            ('XXXXXXXX', 'bbbbbbbb', possible_secrets),
        ]
        for case in cases:
            self.assertEqual(filter_secrets(case[0], case[1], possible_secrets), case[2])

    def test_wierd(self):
        guess = '22-6*3=4'
        clue = 'gbgygygy'
        secrets = ['22-6*3=4', '27-4*6=3', '27-6*4=3']
        expected = ['27-4*6=3']
        actual = filter_secrets(guess, clue, secrets)
        self.assertEqual(expected, actual)
