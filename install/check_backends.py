#!/usr/bin/env python3
"""Startup check for the optional backends: JPEG quality mode (cm=1), SZ3 (cm=8) and the 7zz archive tool."""

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # install/ is one level below the project root
sys.path.insert(0, str(PROJECT_ROOT))


def _reason(exc: Exception) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__


def check_jpeg() -> bool:
    try:
        from compress.turbojpeg_loader import open_turbojpeg

        open_turbojpeg()
    except Exception as exc:  # any failure here disables JPEG quality mode
        print(f"[WARNING] JPEG quality mode (cm=1) is disabled: {_reason(exc)}")
        print("          Run install/setup_linux.sh to build libjpeg-turbo.")
        return False
    print("[OK] JPEG quality mode (cm=1) available")
    return True


def check_sz3() -> bool:
    try:
        # same import as compress/compression_sz3.py
        from pysz import sz, szConfig, szErrorBoundMode  # noqa: F401
    except Exception as exc:
        print(f"[WARNING] SZ3 mode (cm=8) is disabled: {_reason(exc)}")
        print("          Run install/setup_linux.sh to build pysz from install/sz3.")
        return False
    print("[OK] SZ3 mode (cm=8) available")
    return True


def check_7zz() -> bool:
    path = shutil.which("7zz")
    if path is None:
        print("[WARNING] 7zz (7-Zip-zstd) not found: .7z archives use py7zr (slower).")
        print("          Run install/setup_linux.sh to install it into /usr/local/bin.")
        return False
    print(f"[OK] 7zz available: {path}")
    return True


def main() -> int:
    check_jpeg()
    check_sz3()
    check_7zz()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
