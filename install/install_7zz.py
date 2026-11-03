#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Install the newest 7-Zip-zstd (7zz) into /usr/local/bin/7zz.

Source: https://github.com/mcmilk/7-Zip-zstd, Linux x64 / arm64 builds from the latest release.
Called by install/setup_linux.sh. Standard library only.

Exit codes: 0 = 7zz is current or was just installed, 1 = not installed (the printed command finishes it).
"""

import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import urllib.request
import zipfile

REPO = "mcmilk/7-Zip-zstd"
LATEST_API = f"https://api.github.com/repos/{REPO}/releases/latest"
TARGET = "/usr/local/bin/7zz"
CACHE_PATH = os.path.join(os.path.expanduser("~"), ".fidwac", "7zz")
ARCHES = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}


def fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "FIDWAC-v2-installer", "Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return sha256_bytes(handle.read())


def install_to_target(source: str) -> bool:
    """Put source at TARGET by renaming a staged copy over it, so a symlink at TARGET is replaced and never written through.

    Root copies directly; other users need sudo (the password is asked in this terminal).
    """
    staged = TARGET + ".new"
    if os.geteuid() == 0:
        shutil.copyfile(source, staged)
        os.chmod(staged, 0o755)
        os.replace(staged, TARGET)
        return True
    if shutil.which("sudo") is None or subprocess.run(["sudo", "-v"]).returncode != 0:
        return False
    if subprocess.run(["sudo", "install", "-m", "0755", source, staged]).returncode != 0:
        return False
    return subprocess.run(["sudo", "mv", "-f", staged, TARGET]).returncode == 0


def main() -> int:
    if platform.system() != "Linux":
        print(f"[WARN] 7zz: the 7-Zip-zstd Linux build does not run on {platform.system()}; py7zr is used instead.")
        return 0
    machine = platform.machine().lower()
    arch = ARCHES.get(machine)
    if arch is None:
        print(f"[WARN] 7zz: no 7-Zip-zstd Linux build for {machine}; py7zr is used instead.")
        return 0

    asset_name = f"linux-gcc-{arch}.zip"
    try:
        release = json.loads(fetch(LATEST_API))
        tag = release["tag_name"]
        asset = next((a for a in release.get("assets", []) if a.get("name") == asset_name), None)
        if asset is None:
            raise RuntimeError(f"{asset_name} is not in release {tag}")
        blob = fetch(asset["browser_download_url"])
    except Exception as exc:  # offline, GitHub API limit or a changed release layout
        if os.path.isfile(TARGET):
            print(f"[WARN] 7zz: cannot check for a newer release ({exc}); keeping {TARGET}.")
            return 0
        print(f"[WARN] 7zz: cannot download the release ({exc}).")
        print(f"       Get {asset_name} from https://github.com/{REPO}/releases/latest and install 7zz by hand.")
        return 1

    digest = asset.get("digest") or ""
    if digest.startswith("sha256:"):
        if sha256_bytes(blob) != digest[len("sha256:"):]:
            print("[ERROR] 7zz: the download does not match the SHA-256 published by GitHub. Nothing was installed.")
            return 1
    else:
        print("[WARN] 7zz: GitHub published no SHA-256 for this file; the checksum was not verified.")

    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        binary = archive.read("7zz")
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as handle:
        handle.write(binary)
    os.chmod(CACHE_PATH, 0o755)

    if os.path.isfile(TARGET) and not os.path.islink(TARGET) and sha256_file(TARGET) == sha256_bytes(binary):
        print(f"[OK] 7zz {tag} is already installed: {TARGET}")
        return 0

    if os.path.islink(TARGET):
        print(f"[INFO] {TARGET} is a symlink to {os.readlink(TARGET)}; it is replaced by the 7-Zip-zstd binary.")
    print(f"Installing 7zz {tag} to {TARGET} (sudo may ask for your password) ...")
    if install_to_target(CACHE_PATH):
        print(f"[OK] 7zz {tag} installed: {TARGET}")
        return 0
    print("[WARN] 7zz was not installed (sudo did not succeed). Run this command to finish:")
    print(f"       sudo install -m 0755 {CACHE_PATH} {TARGET}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
