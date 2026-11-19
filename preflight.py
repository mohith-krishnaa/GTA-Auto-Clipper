"""Check whether the local machine has the basic tools needed by GTA Auto Clipper.

This check is intentionally read-only: it never downloads models, contacts APIs,
prints credential values, or starts recording.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil

PYTHON_MODULES = {
    "numpy": "numpy",
    "faster-whisper": "faster_whisper",
    "yt-dlp": "yt_dlp",
    "requests": "requests",
    "OpenCV": "cv2",
}


def inspect_environment(environ=None, which=shutil.which, find_spec=importlib.util.find_spec):
    """Return structured checks without exposing environment variable values."""
    env = os.environ if environ is None else environ
    checks = []

    for label, module in PYTHON_MODULES.items():
        try:
            available = find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        checks.append({
            "name": f"Python package: {label}",
            "status": "ok" if available else "missing",
            "required": label in {"numpy", "faster-whisper", "yt-dlp"},
            "detail": "installed" if available else f"Install with: python -m pip install -r requirements.txt",
        })

    ffmpeg = which("ffmpeg")
    checks.append({
        "name": "FFmpeg",
        "status": "ok" if ffmpeg else "missing",
        "required": True,
        "detail": ffmpeg or "Install FFmpeg and add it to PATH.",
    })

    runtime = next((name for name in ("deno", "node", "bun") if which(name)), None)
    checks.append({
        "name": "JavaScript runtime",
        "status": "ok" if runtime else "warning",
        "required": False,
        "detail": runtime or "Install Node.js or Deno for reliable YouTube extraction.",
    })

    configured = bool(env.get("YOUTUBE_API_KEY", "").strip())
    checks.append({
        "name": "YouTube API key",
        "status": "configured" if configured else "optional",
        "required": False,
        "detail": "Set (value hidden)" if configured
                  else "Not set; live search/chat features will be unavailable.",
    })
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a failure code when a required dependency is missing",
    )
    args = parser.parse_args(argv)

    checks = inspect_environment()
    print("GTA Auto Clipper — environment check")
    print("=" * 39)
    for check in checks:
        print(f"[{check['status'].upper():10}] {check['name']}: {check['detail']}")

    missing = [item for item in checks if item["required"] and item["status"] != "ok"]
    if missing:
        print(f"\n{len(missing)} required check(s) failed.")
        return 1 if args.strict else 0
    print("\nRequired local dependencies look ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
