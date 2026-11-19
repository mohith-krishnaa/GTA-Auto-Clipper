import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import preflight


class PreflightTests(unittest.TestCase):
    def test_missing_required_dependencies_are_reported_without_crashing(self):
        checks = preflight.inspect_environment(
            environ={},
            which=lambda name: None,
            find_spec=lambda name: None,
        )
        by_name = {item["name"]: item for item in checks}
        self.assertEqual(by_name["FFmpeg"]["status"], "missing")
        self.assertEqual(by_name["Python package: numpy"]["status"], "missing")
        self.assertEqual(by_name["JavaScript runtime"]["status"], "warning")
        self.assertEqual(by_name["YouTube API key"]["status"], "optional")

    def test_configured_credentials_are_never_printed_or_returned(self):
        checks = preflight.inspect_environment(
            environ={"YOUTUBE_API_KEY": "do-not-leak-this"},
            which=lambda name: "C:/tools/" + name if name in {"ffmpeg", "node"} else None,
            find_spec=lambda name: object(),
        )
        key_check = next(item for item in checks if item["name"] == "YouTube API key")
        self.assertEqual(key_check["status"], "configured")
        self.assertNotIn("do-not-leak-this", repr(checks))

    def test_main_returns_failure_in_strict_mode_when_required_dependency_is_missing(self):
        with patch("preflight.inspect_environment", return_value=[
            {"name": "FFmpeg", "status": "missing", "required": True, "detail": "install"},
        ]):
            self.assertEqual(preflight.main(["--strict"]), 1)

    def test_main_is_non_failing_without_strict_when_required_check_is_missing(self):
        with patch("preflight.inspect_environment", return_value=[
            {"name": "FFmpeg", "status": "missing", "required": True, "detail": "install"},
        ]):
            self.assertEqual(preflight.main([]), 0)

    def test_main_succeeds_when_required_checks_pass_even_without_optional_api_key(self):
        with patch("preflight.inspect_environment", return_value=[
            {"name": "FFmpeg", "status": "ok", "required": True, "detail": "available"},
            {"name": "YouTube API key", "status": "optional", "required": False, "detail": "not set"},
        ]):
            self.assertEqual(preflight.main([]), 0)


if __name__ == "__main__":
    unittest.main()
