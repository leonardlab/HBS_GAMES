#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 15:26:19 2022

@author: kate
"""

import unittest

from games.models import ModelA


def test_num_params(self):
    tests = [(100, 100)]

    for given, expected in tests:
        with self.subTest(given=given):

            free_parameters = [15] * 6
            inputs = [50] * 2
            input_ligand = 1000
            model = ModelA(free_parameters, inputs, input_ligand)

            # solve model and plot results
            solution, t = model.solve_single()

            found = len(t)
            self.assertEqual(found, expected)


test_num_params()


class TestGAMES(unittest.TestCase):
    # Test definition of number of parameters for each model

    def test_num_params(self):
        tests = [(100, 100)]

        for given, expected in tests:
            with self.subTest(given=given):

                free_parameters = [15] * 6
                inputs = [50] * 2
                input_ligand = 1000
                model = ModelA(free_parameters, inputs, input_ligand)

                # solve model and plot results
                solution, t = model.solve_single()

                found = len(t)
                self.assertEqual(found, expected)
