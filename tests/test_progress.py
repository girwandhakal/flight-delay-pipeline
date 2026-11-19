"""Stage messages must identify completion and preserve failures."""

from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

from scripts.progress import stage


class ProgressTests(unittest.TestCase):
    def test_failure_reports_stage_and_reraises_original_error(self):
        for error in (ValueError("invalid observations"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                output = io.StringIO()
                with redirect_stdout(output), patch("scripts.progress.time.monotonic", side_effect=[10, 13]):
                    with self.assertRaises(type(error)) as caught:
                        with stage("Clean weather"):
                            raise error
                self.assertIs(caught.exception, error)
                self.assertIn(
                    f"FAILED: Clean weather after 3.0s ({type(error).__name__}: {error})",
                    output.getvalue(),
                )
                self.assertNotIn("DONE:", output.getvalue())
