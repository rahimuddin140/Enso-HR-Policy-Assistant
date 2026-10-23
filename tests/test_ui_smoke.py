"""Smoke test for loading the application entrypoint and submitting a question."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


class TestApplicationSmoke(unittest.TestCase):
    def test_app_cli_status_and_question_submission(self) -> None:
        import os

        env = dict(os.environ)
        env["OFFLINE_MODE"] = "1"
        status_proc = subprocess.run(
            [sys.executable, str(ROOT_DIR / "app.py"), "--status"],
            cwd=str(ROOT_DIR),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        status_data = json.loads(status_proc.stdout)
        self.assertIn("indexed_chunks", status_data)
        self.assertGreater(status_data["indexed_chunks"], 0)

        q_proc = subprocess.run(
            [
                sys.executable,
                str(ROOT_DIR / "app.py"),
                "--question",
                "How many days of Privilege Leave (PL) does a permanent employee get per year?",
            ],
            cwd=str(ROOT_DIR),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        q_data = json.loads(q_proc.stdout)
        self.assertTrue(q_data["found"])
        self.assertIn("12 days", q_data["answer"])
        self.assertEqual(q_data["sources"][0]["page"], 60)


if __name__ == "__main__":
    unittest.main()
