#!/usr/bin/env python3

import unittest
from sandbox.fibonacci import Fibonacci


class TestFibonacci(unittest.TestCase):
    # Test Fibonacci calculate function

    def test_fibonacci_first_seven_unittest(self):
        tests = [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (5, 5), (6, 8)]
        for given, expected in tests:
            with self.subTest(given=given):
                found = Fibonacci().calculate(given)
                self.assertEqual(found, expected)
