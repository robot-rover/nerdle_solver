import unittest

from nerdle_solver.equation import validate, parse


class TestEquationParse(unittest.TestCase):

    def test_succeed(self):
        cases = [
            ('3+22*5=13', [3, '+', 22, '*', 5, '=', 13]),
            ('10+20=30', [10, '+', 20, '=', 30]),
            ('1+2*3+4=11', [1, '+', 2, '*', 3, '+', 4, '=', 11])
        ]
        for case in cases:
            self.assertEqual(parse(case[0]), case[1], str(case))

    def test_fail(self):
        cases = [
            '3a22*5=1',
            '19&1=1'
        ]
        for case in cases:
            with self.assertRaises(RuntimeError, msg=str(case)):
                parse(case)


class TestEquationCalc(unittest.TestCase):

    def test_succeed(self):
        cases = [
            ([3, '+', 22, '*', 5, '=', 113]),
            ([10, '+', 20, '=', 30]),
            ([1, '+', 2, '*', 3, '+', 4, '=', 11])
        ]
        for case in cases:
            self.assertTrue(validate(case), str(case))

    def test_fail(self):
        cases = [
            ([3, '+', 22, '*', 5, '=', 10]),
            ([10, '-', 20, '=', 30]),
            ([10, '-', '+', 20, '=', 30]),
            (['-', '+', 20, '=', 30]),
            (['-', '+', 20, 30]),
            (['-', '+', 20, 30, '=']),
            ([1, 2, '*', 3, '+', 4, '=', 11])
        ]
        for case in cases:
            self.assertFalse(validate(case), str(case))
