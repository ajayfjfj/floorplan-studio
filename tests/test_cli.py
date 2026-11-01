import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from floorplan.cli import main


class CliTests(unittest.TestCase):
    def invoke(self, arguments):
        output, errors = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["floorplan", *arguments]):
            with redirect_stdout(output), redirect_stderr(errors):
                with self.assertRaises(SystemExit) as exit_status:
                    main()
        return exit_status.exception.code, output.getvalue(), errors.getvalue()

    def test_root_help_exits_cleanly(self):
        code, output, errors = self.invoke(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("usage:", output)
        self.assertEqual(errors, "")

    def test_each_subcommand_help_exits_cleanly(self):
        for command in ["prepare", "train", "infer", "evaluate"]:
            with self.subTest(command=command):
                code, output, errors = self.invoke([command, "--help"])
                self.assertEqual(code, 0)
                self.assertIn(f"floorplan {command}", output)
                self.assertEqual(errors, "")

    def test_invalid_arguments_exit_nonzero(self):
        cases = [
            [],
            ["not-a-command"],
            ["infer"],
            ["prepare", "--raw", "unused", "--output", "unused", "--not-an-option"],
            [
                "evaluate",
                "--data",
                "unused",
                "--checkpoint",
                "unused",
                "--output",
                "unused",
                "--split",
                "train",
            ],
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                code, output, errors = self.invoke(arguments)
                self.assertNotEqual(code, 0)
                self.assertEqual(output, "")
                self.assertIn("error:", errors)


if __name__ == "__main__":
    unittest.main()
