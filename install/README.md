# FIDWAC v2 Installation Guide

This guide contains complete installation instructions for Windows (WSL2), Linux, and macOS, including Conda setup and troubleshooting.

## System Requirements

### Operating Systems

- Windows 10 or Windows 11 with WSL2 (Windows Subsystem for Linux 2)
- Linux (native):
  - Debian/Ubuntu 20.04 LTS or newer
  - Fedora 35+
  - RHEL 8+
  - Arch Linux
  - Other distributions with Python 3.9+
- macOS (Intel or Apple Silicon) with Python 3.10+

### Required Components

On Windows:
- WSL2 installed and configured
- Any Linux distribution (Debian recommended)

On Linux (WSL2 or native):
- Python 3.9 or newer
- pip (Python package manager)
- Basic build tools: build-essential or equivalent
- For full functionality (JPEG compression, GDAL, etc.):
  - libgdal-dev (Debian/Ubuntu), gdal-devel (Fedora/RHEL), gdal (Arch)
  - cmake and nasm (Debian/Ubuntu, Fedora/RHEL, Arch). nasm enables SIMD in libjpeg-turbo, which the installer builds itself from install/turbojpeg
  - llvm-dev (Debian/Ubuntu), llvm-devel (Fedora/RHEL), llvm (Arch)
  - 7zz (7-Zip-ZSTD) is not a distribution package: `install/setup_linux.sh` installs the newest release to `/usr/local/bin/7zz` (sudo needed, see [7-Zip-ZSTD](#7-zip-zstd-7zz-7z-archives))

On macOS:
- Python 3.10+ (Conda or python.org installer recommended); the web GUI needs Flask from `install/requirements.txt`
- Homebrew packages for full functionality:
  - gdal
  - jpeg-turbo
  - sevenzip (provides `7zz`; the upstream 7-Zip build, the zstd fork has no macOS build)
- Xcode Command Line Tools: xcode-select --install

RAM: minimum 2 GB, recommended 4+ GB for large rasters
Disk space: minimum 500 MB (+ space for processed raster files)

## Installation on Windows (WSL2)

### Step 1: Prepare WSL2

If you do not have WSL2 yet, install it:

```cmd
wsl --install
```

Restart your computer. After restart, open your Linux distribution from the Start menu and set up your username and password.

Verify WSL2 is working:

```cmd
wsl --list --verbose
```

You should see your distribution listed with version 2.

### Step 2: Install FIDWAC v2

Open Command Prompt and navigate to the project directory:

```cmd
cd path\to\FIDWAC_v2
```

Run the installation script:

```cmd
install\setup_windows.bat
```

The script will:
1. Check available WSL2 distributions
2. Ask you to select a distribution (or install a new one)
3. Ask you to select/create a Linux user
4. Install Python 3 and required system packages
5. Create a Python virtual environment at /home/user/.fidwac/venv
6. Install Python dependencies from install/requirements.txt

Expected time: 5-15 minutes (depends on internet speed and disk).

### Step 3: Run the Application

After successful installation:

```cmd
run_windows.bat
```

The FIDWAC v2 web GUI starts and prints its address. Open **http://127.0.0.1:5010/** in your browser.

## Installation on Linux (native)

### Step 1: Prepare Your System

On Debian/Ubuntu:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
```

On Fedora/RHEL:

```bash
sudo dnf install -y python3 python3-pip
```

On Arch:

```bash
sudo pacman -S python python-pip
```

### Step 2: Install FIDWAC v2

Navigate to the project directory:

```bash
cd /path/to/FIDWAC_v2
```

Run the installation script:

```bash
bash install/setup_linux.sh
```

The script will install:
1. A Python virtual environment at ~/.fidwac/venv
2. Required system packages (GDAL, LLVM, cmake, nasm, zenity — zenity draws the interface chooser of run_linux.sh)
3. Python dependencies from install/requirements.txt (includes Flask for the web GUI)
4. libjpeg-turbo (JPEG quality mode, `cm=1`), built from `install/turbojpeg` into `~/.fidwac/venv/libjpeg-turbo`
5. 7zz (7-Zip-ZSTD), the newest release from GitHub, installed to `/usr/local/bin/7zz` (sudo)

If system package installation fails, the script will display instructions on how to install them manually for your distribution.

### Step 3: Run the Application

```bash
bash run_linux.sh
```

The FIDWAC v2 web GUI starts and prints its address. Open **http://127.0.0.1:5010/** in your browser (`bash run_linux.sh --port 8080` uses another port).

## Installation on macOS

### Option A: Virtual environment + Homebrew

1. Install Apple command-line tools:

```bash
xcode-select --install
```

2. Install system packages:

```bash
brew install gdal jpeg-turbo sevenzip
```

3. Create and activate a virtual environment:

```bash
cd /path/to/FIDWAC_v2
python3 -m venv ~/.fidwac/venv
source ~/.fidwac/venv/bin/activate
python -m pip install --upgrade pip
pip install -r install/requirements.txt
```

4. Verify the environment:

```bash
python3 install/verify_install.py --require-web --require-turbojpeg
```

5. Run the web GUI or the CLI:

```bash
python3 app.py
```

The web GUI then listens on http://127.0.0.1:5010/. Flask comes from `install/requirements.txt`; if `import flask` fails, run `pip install -r install/requirements.txt` again.

## Installation with Conda

The repository includes a ready-to-use Conda environment for macOS, Linux, and WSL2. It does not include a 7z program: `.7z` archives use `7zz` from `install/setup_linux.sh` when it is installed, and `py7zr` otherwise.

```bash
cd /path/to/FIDWAC_v2
conda env create -f install/environment.yml
conda activate fidwac
python3 install/verify_install.py --require-web --require-turbojpeg
```

## SZ3 backend (`cm=8`, `--method sz3`)

The SZ3 compressor is built from the source tree in `install/sz3` (SZ3 3.3.2), not downloaded. `install/setup_linux.sh` does this automatically. To do it by hand from the project root, with the FIDWAC virtual environment active (the build runs in a temporary copy, so no build files are left in `install/sz3`):

```bash
SZ3_TMP="$(mktemp -d)"
cp -r install/sz3 "$SZ3_TMP/sz3"
SZ3_SRC="$SZ3_TMP/sz3" pip install --force-reinstall --no-deps "$SZ3_TMP/sz3/tools/pysz"
rm -rf "$SZ3_TMP"
python3 install/verify_install.py --require-sz3
```

Requirements: `cmake` 3.24 or newer, a C++17 compiler (`g++`), and zstd development files (`libzstd-dev` on Debian/Ubuntu, `libzstd-devel` on Fedora). If zstd is not installed, CMake builds the bundled copy in `install/sz3/tools/zstd`. Without the SZ3 backend the web GUI and the `dct` and `pred` methods still work; only `--method sz3` is unavailable.

---

## libjpeg-turbo (`cm=1`, JPEG quality mode)

The source is in `install/turbojpeg`. `install/setup_linux.sh` asks GitHub for the newest libjpeg-turbo release. If `install/turbojpeg` is older, the script downloads that release and replaces the folder; if GitHub cannot be reached, it builds the copy already there. The build runs in a temporary directory and installs into `~/.fidwac/venv/libjpeg-turbo` (no `sudo`). With `nasm` present the SIMD code paths are built, otherwise libjpeg-turbo uses plain C. The application loads that build by explicit path (`compress/turbojpeg_loader.py`), so an older system copy (2.x) is not used.

Check it with: `python3 install/verify_install.py --require-turbojpeg`

Without it the web GUI and the `dct`, `pred` and `sz3` methods still work; only JPEG quality mode is unavailable.

---

## 7-Zip-ZSTD (`7zz`, `.7z` archives)

`.7z` output uses the `7zz` command from the 7-Zip-ZSTD project, https://github.com/mcmilk/7-Zip-zstd. It is not a distribution package. `install/setup_linux.sh` runs `install/install_7zz.py`, which:

1. asks GitHub for the newest release and downloads its Linux build for your CPU (`linux-gcc-x64.zip` or `linux-gcc-arm64.zip`);
2. checks the file against the SHA-256 that GitHub publishes for it;
3. installs the binary to `/usr/local/bin/7zz`. This needs `sudo`, and the password is asked in the terminal. If `7zz` is already the newest build, nothing is changed.

If `/usr/local/bin/7zz` is a symlink (for example to another 7-Zip), it is replaced by the 7-Zip-ZSTD file; the link target is not modified. Without sudo, the script keeps the download in `~/.fidwac/7zz` and prints the `sudo install` command to finish. To update later, run `python3 install/install_7zz.py` again.

Without `7zz` the program writes `.7z` archives with `py7zr`, which works but is slower. The startup check and `install/verify_install.py` report which archive tool is used. macOS: `brew install sevenzip` gives the upstream `7zz`.

---

## Interpolation (IDW, `interpolate.py`)

The interpolation packages (`laspy`, `lazrs`, `pyshp`, `matplotlib`) are listed in `install/requirements.txt` and `install/environment.yml`, so the normal installation includes them. The geoid model `interpolation/geoid/PL_geod.7z` is part of the repository. `install/verify_install.py` checks the packages, loads the geoid model and runs a small IDW test; a problem in this part is reported as a warning and does not fail the compression checks. Usage is described in the main README: [Interpolation (IDW)](../README_EN.md#interpolation-idw).

## Third-party components, links and licenses

The complete list, including Python packages, is in the main README: [Third-Party Components, Links and Licenses](../README_EN.md#third-party-components-links-and-licenses). These are the components that the setup script installs or builds:

| Component | GitHub | License |
| --- | --- | --- |
| 7-Zip-ZSTD (`7zz`) | https://github.com/mcmilk/7-Zip-zstd | LGPL-2.1-or-later (some files LGPL with unRAR restriction, some BSD-3-Clause) |
| SZ3 (vendored in `install/sz3`) and pysz | https://github.com/szcompressor/SZ3 | BSD-3-Clause |
| libjpeg-turbo (vendored in `install/turbojpeg`) | https://github.com/libjpeg-turbo/libjpeg-turbo | IJG License and Modified BSD-3-Clause (see its `LICENSE.md`) |
| zstd (`libzstd-dev`, `libzstd-devel`) | https://github.com/facebook/zstd | BSD-3-Clause or GPL-2.0 (dual license) |
| NASM | https://github.com/netwide-assembler/nasm | BSD-2-Clause |
| CMake | https://github.com/Kitware/CMake | BSD-3-Clause |
| GDAL (`libgdal-dev`, `gdal-devel`, `gdal`) | https://github.com/OSGeo/gdal | MIT-style (components listed in `LICENSE.TXT`) |
| LLVM (`llvm-dev`, `llvm-devel`, `llvm`) | https://github.com/llvm/llvm-project | Apache-2.0 with LLVM Exceptions |
| ZSIP/FIDWaC (IDW algorithm, adapted in `interpolation/`) | https://github.com/ZSIP/FIDWaC | MIT |
| Geoid model PL (`interpolation/geoid/PL_geod.7z`) | Optional geoid correction in `interpolate.py` | — (data, not a repository) | To be confirmed by the author: source and licence not stated in the archive |

---

## Installation Troubleshooting

At startup `run_linux.sh` and `run_windows.bat` report whether the JPEG (`cm=1`) and SZ3 (`cm=8`) backends are available (`install/check_backends.py`).

### Issue: "Virtual environment not found"

Cause: The run script could not find the installed virtual environment.

Solution:
1. Make sure you ran setup_windows.bat (Windows) or setup_linux.sh (Linux)
2. Wait for installation to complete
3. Try again

### Issue: Missing system packages (warning during installation)

Cause: Some system packages could not be installed automatically.

Solution: The script will display instructions on how to install them manually for your distribution. Follow those instructions and run the recommended sudo apt install, sudo dnf install, etc. commands.

### Issue: "PyTurboJPEG not available"

Cause: libjpeg-turbo could not be built from `install/turbojpeg` (cmake missing, or a compile error; see `/tmp/fidwac_libjpeg_build.log`).

Solution: JPEG quality mode is optional. Install cmake (and nasm for SIMD), then rerun `install/setup_linux.sh`.

### Issue: "No module named flask"

Cause: The virtual environment was created before Flask was added to `install/requirements.txt`, or the environment is not active.

Solution:

```bash
source ~/.fidwac/venv/bin/activate
pip install -r install/requirements.txt
python3 install/verify_install.py --require-web --require-turbojpeg
```

### Issue: 7zz not found (archives written with py7zr)

Cause: `/usr/local/bin/7zz` is missing, because sudo was not available during setup or GitHub could not be reached.

Solution: If the installer kept the download, finish with `sudo install -m 0755 ~/.fidwac/7zz /usr/local/bin/7zz`. Otherwise rerun `bash install/setup_linux.sh` with sudo available. Check with `python3 install/verify_install.py`.
