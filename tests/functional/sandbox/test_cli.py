#!/usr/bin/env python3
import unittest
from click.testing import CliRunner
from sandbox.cli import cli


class TestCLI(unittest.TestCase):
    """Functional tests for sandbox CLI."""

    def test_cli_prints_nth_fibonacci_number_and_exits_0(self):
        # User passes number to program.
        commands = ["10"]

        # Program calls CLI.
        runner = CliRunner()
        result = runner.invoke(cli, commands)

        # Check exit code
        self.assertEqual(result.exit_code, 0)

        # ...and output value.
        self.assertEqual(
            result.output.strip(), "[1, 1, 2, 3, 5, 8, 13, 21, 34, 55]\n55", "first check failed"
        )
