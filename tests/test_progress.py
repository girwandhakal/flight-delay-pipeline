"""Stage messages must identify completion and preserve failures."""

from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

from scripts.progress import stage


class ProgressTests(unittest.TestCase):
    def test_failure_reports_stage_and_reraises_original_error(self):
        output = io.StringIO()
        error = ValueError("invalid observations")
        with redirect_stdout(output), patch("scripts.progress.time.monotonic", side_effect=[10, 13]):
            with self.assertRaises(ValueError) as caught:
                with stage("Clean weather"):
                    raise error
        self.assertIs(caught.exception, error)
        self.assertIn("FAILED: Clean weather after 3.0s (ValueError: invalid observations)", output.getvalue())
        self.assertNotIn("DONE:", output.getvalue())
