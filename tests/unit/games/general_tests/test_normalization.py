#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 15:26:19 2022

@author: kate
"""

import unittest
from games.config.experimental_data import normalize_data_by_maximum_value


class TestNormalizeDataByMaximumValue(unittest.TestCase):
    def test_normalize_data_by_maximum_value(self):
        dataID = "ligand dose response"
        tests = [
            (([2, 5, 10], None), ([0.2, 0.5, 1.0], [0])),
            (([2, 5, 10], [2, 5, 10]), ([0.2, 0.5, 1.0], [0.2, 0.5, 1.0])),
        ]
        for (given_1, given_2), (expected_1, expected_2) in tests:
            if given_2 == None:
                found_1, found_2 = normalize_data_by_maximum_value(given_1, dataID)
            else:
                found_1, found_2 = normalize_data_by_maximum_value(given_1, dataID, given_2)
            self.assertEqual(found_1, expected_1)
            self.assertEqual(found_2, expected_2)


if __name__ == "__main__":
    unittest.main()
