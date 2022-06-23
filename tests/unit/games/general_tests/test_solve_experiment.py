#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 15:26:19 2022

@author: kate
"""

import unittest
from games.models.synTF_chem import synTF_chem


class TestSolveExperiment(unittest.TestCase):
    def test_solve_experiment_synTF_chem(self):
        model = synTF_chem(parameters=[1] * 6)
        tests = [([1] * 10)]
        tests = [
            (([1] * 11, "ligand dose response"), 11),
            (([1] * 27, "ligand dose response and DBD dose response"), 27),
        ]
        for (given_1, given_2), expected in tests:
            solution = model.solve_experiment(given_1, given_2)
            found = len(solution)
        self.assertEqual(found, expected)


if __name__ == "__main__":
    unittest.main()
