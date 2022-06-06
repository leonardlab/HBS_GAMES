#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 15:26:19 2022

@author: kate
"""

import unittest

from games.modules import Modules
from games.set_model import model


class TestGames(unittest.TestCase):
    def test_chi_sq(self):
        parameters1 = [2] * 6
        parameters2 = [1] * 6
        tests = [(parameters1, 1), (parameters2, 1)]

        for given, expected in tests:
            with self.subTest(given=given):
                model.parameters = given
                solutions, chi_sq, r_sq = Modules.test_single_parameter_set()
                found = len(chi_sq)
                self.assertEqual(found, expected)
