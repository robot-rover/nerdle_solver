import unittest

from ..clues import generate_clue

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
        ]
        for case in cases:
            self.assertEqual(generate_clue(case[0], case[1]), case[2])