#!/bin/bash
# FIDWAC v2 - Linux Setup Script (WSL2 or native Linux)
# Sets up virtual environment and installs dependencies (no CUDA)

set +e  # Do not exit on error (we handle errors manually)

# Initialize variables
FAILED_DEPS=()
PKG_MANAGER=""

echo "========================================"
echo "FIDWAC v2 Linux Installer"
echo "========================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Project path: $PROJECT_ROOT"
echo ""

# Detect execution environment
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "[ERROR] This installer supports only Linux and WSL2"
    exit 1
fi

if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "[OK] WSL2 environment detected"
else
    echo "[OK] Native Linux environment detected"
fi

echo ""

# Check if Python 3 is installed
echo "Checking Python 3..."
if ! command -v python3 &> /dev/null; then
    echo "[WARNING] Python 3 is not installed"
    echo "Installing Python 3 and related packages..."
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install Python 3"
        exit 1
    fi
    echo "[OK] Python 3 installed"
fi

PYTHON_VERSION=$(python3 --version)
echo "[OK] $PYTHON_VERSION"
echo ""

# Check if virtual environment exists
# Store venv in user's home directory (.fidwac), not in project directory
VENV_DIR="$HOME/.fidwac/venv"

# A venv is usable only if its activate script exists, pip runs inside it,
# and it was built with a Python new enough for the pinned requirements.
PY_MIN_MAJOR=3
PY_MIN_MINOR=12

venv_is_healthy() {
    [ -f "$VENV_DIR/bin/activate" ] && "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1 \
        && "$VENV_DIR/bin/python" -c "import sys; sys.exit(0 if sys.version_info >= ($PY_MIN_MAJOR, $PY_MIN_MINOR) else 1)" 2>/dev/null
}

# Pick the first interpreter that satisfies PY_MIN_MAJOR.PY_MIN_MINOR (newest first).
find_python() {
    local cand
    for cand in python3.14 python3.13 python3.12 python3; do
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import sys; sys.exit(0 if sys.version_info >= ($PY_MIN_MAJOR, $PY_MIN_MINOR) else 1)" 2>/dev/null; then
            echo "$cand"
            return 0
        fi
    done
    return 1
}

# Build the venv from scratch. Debian/Ubuntu ship python3 without ensurepip
# unless python3-venv is installed, so install it and retry once.
create_venv() {
    local py
    rm -rf "$VENV_DIR"
    mkdir -p "$(dirname "$VENV_DIR")"

    if ! py="$(find_python)" && command -v apt-get &> /dev/null; then
        echo "[WARNING] Python ${PY_MIN_MAJOR}.${PY_MIN_MINOR}+ not found; trying to install python3.12 via apt..."
        sudo apt-get update -qq
        sudo apt-get install -y python3.12 python3.12-venv >/dev/null 2>&1 || true
        py="$(find_python)" || py=""
    fi
    if [ -z "$py" ]; then
        echo "[ERROR] Python ${PY_MIN_MAJOR}.${PY_MIN_MINOR}+ is required (numpy 2.5 needs it)."
        echo "Install it (e.g. sudo apt install python3.12 python3.12-venv) and rerun this script."
        return 1
    fi
    echo "Using $("$py" --version) for the virtual environment"

    if "$py" -m venv "$VENV_DIR" && venv_is_healthy; then
        return 0
    fi

    rm -rf "$VENV_DIR"
    if command -v apt-get &> /dev/null; then
        echo "[WARNING] Python venv/pip support is missing; installing ${py}-venv..."
        sudo apt-get update -qq
        sudo apt-get install -y "${py}-venv" python3-pip
        "$py" -m venv "$VENV_DIR" && venv_is_healthy && return 0
        rm -rf "$VENV_DIR"
    fi
    return 1
}

# Remove a leftover or outdated venv so it gets rebuilt below.
if [ -d "$VENV_DIR" ] && ! venv_is_healthy; then
    echo "[WARNING] Existing virtual environment is broken or uses Python older than ${PY_MIN_MAJOR}.${PY_MIN_MINOR}: $VENV_DIR"
    echo "Removing it and creating a new one..."
    rm -rf "$VENV_DIR"
fi

if [ -d "$VENV_DIR" ]; then
    echo "[INFO] Virtual environment already exists: $VENV_DIR"
    echo "Do you want to recreate it? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment."
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    if ! create_venv; then
        echo "[ERROR] Failed to create a working virtual environment in $VENV_DIR"
        echo "Install python3-venv and python3-pip manually, then rerun this script."
        exit 1
    fi
    echo "[OK] Virtual environment created: $VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
if ! source "$VENV_DIR/bin/activate"; then
    echo "[ERROR] Could not activate virtual environment: $VENV_DIR"
    exit 1
fi

# Always run pip through the venv's interpreter, so a missing 'pip' wrapper
# script cannot break the installation.
pip() {
    "$VENV_DIR/bin/python" -m pip "$@"
}

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q || echo "[WARN] pip upgrade failed; continuing with the installed version."

# Detect or install CUDA toolkit and GPU runtime for Numba/CUDA.
detect_cuda_root() {
    local cand
    local seen=()
    for cand in \
        "${CUDA_HOME:-}" \
        "${NV:-}" \
        "$HOME"/.fidwac/venv/lib/python*/site-packages/nvidia/cu* \
        "/usr/local/cuda" \
        "/usr/local/cuda-13" \
        "/usr/local/cuda-12" \
        "/usr/local/cuda-11" \
        "/usr/local/cuda-12.8" \
        "/usr/local/cuda-13.1" \
        "/opt/cuda"; do
        if [ -n "$cand" ] && [ -x "$cand/bin/nvcc" ]; then
            echo "$cand"
            return 0
        fi
    done
    if command -v nvcc >/dev/null 2>&1; then
        echo "$(dirname "$(dirname "$(command -v nvcc)")")"
        return 0
    fi
    return 1
}

ensure_cuda_runtime() {
    if detect_cuda_root >/dev/null 2>&1; then
        return 0
    fi

    echo "[INFO] CUDA toolkit not detected; attempting to install the NVIDIA CUDA runtime packages into the venv."
    pip install --quiet --upgrade \
        nvidia-cuda-runtime-cu12 \
        nvidia-cuda-nvcc-cu12 \
        nvidia-cuda-cupti-cu12 2>/dev/null || true

    if detect_cuda_root >/dev/null 2>&1; then
        echo "[OK] CUDA toolkit discovered after installation."
        return 0
    fi

    return 1
}

echo ""

# Install system dependencies
echo "Checking system dependencies..."

# Detect package manager and distribution
PKG_MANAGER=""
DISTRO=""
FAILED_DEPS=()

if command -v apt-get &> /dev/null; then
    PKG_MANAGER="apt"
    DISTRO="Debian/Ubuntu"
    GDAL_PKG="libgdal-dev"
    NASM_PKG="nasm"
    LLVM_PKG="llvm-dev"

    LLVM_PKG="llvm-dev"
ZENITY_PKG="zenity"
    CMAKE_PKG="cmake"
    BUILD_PKG="build-essential"
    PKG_CHECK_CMD="dpkg -l | grep -q"
elif command -v dnf &> /dev/null; then
    PKG_MANAGER="dnf"
    DISTRO="Fedora/RHEL 8+"
    GDAL_PKG="gdal-devel"
    NASM_PKG="nasm"
    LLVM_PKG="llvm-devel"

    LLVM_PKG="llvm-devel"
ZENITY_PKG="zenity"
    CMAKE_PKG="cmake"
    BUILD_PKG="gcc-c++ make"
    PKG_CHECK_CMD="rpm -q"
elif command -v yum &> /dev/null; then
    PKG_MANAGER="yum"
    DISTRO="RHEL 7/CentOS"
    GDAL_PKG="gdal-devel"
    NASM_PKG="nasm"
    LLVM_PKG="llvm-devel"

    LLVM_PKG="llvm-devel"
ZENITY_PKG="zenity"
    CMAKE_PKG="cmake"
    BUILD_PKG="gcc-c++ make"
    PKG_CHECK_CMD="rpm -q"
elif command -v pacman &> /dev/null; then
    PKG_MANAGER="pacman"
    DISTRO="Arch Linux"
    GDAL_PKG="gdal"
    NASM_PKG="nasm"
    LLVM_PKG="llvm"

    LLVM_PKG="llvm"
ZENITY_PKG="zenity"
    CMAKE_PKG="cmake"
    BUILD_PKG="base-devel"
    PKG_CHECK_CMD="pacman -Q"
else
    echo "[WARNING] Package manager not detected. Skipping system dependencies."
    echo "Please install manually: GDAL, LLVM, cmake, nasm, zenity (7zz: see install/README.md)"
    echo ""
fi

if [ -n "$PKG_MANAGER" ]; then
    echo "[INFO] Detected: $DISTRO (using $PKG_MANAGER)"
    echo ""
    
    MISSING_DEPS=()
    
    # Check packages
    if ! eval "$PKG_CHECK_CMD '$GDAL_PKG' > /dev/null 2>&1"; then
        MISSING_DEPS+=("$GDAL_PKG")
    fi
    if ! eval "$PKG_CHECK_CMD '$CMAKE_PKG' > /dev/null 2>&1"; then
        MISSING_DEPS+=("$CMAKE_PKG")
    fi
    if ! eval "$PKG_CHECK_CMD '$NASM_PKG' > /dev/null 2>&1"; then
        MISSING_DEPS+=("$NASM_PKG")
    fi
    if ! eval "$PKG_CHECK_CMD '$LLVM_PKG' > /dev/null 2>&1"; then
        MISSING_DEPS+=("$LLVM_PKG")
    fi
    if ! eval "$PKG_CHECK_CMD '$ZENITY_PKG' > /dev/null 2>&1"; then
        MISSING_DEPS+=("$ZENITY_PKG")
    fi
    
    if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
        echo "[WARNING] Missing system dependencies: ${MISSING_DEPS[*]}"
        echo "Attempting to install..."
        echo ""
        
        echo "Installing them needs sudo: enter your password when asked."
        if sudo -v 2>/dev/null; then
            case "$PKG_MANAGER" in
                apt)
                    sudo apt update -qq
                    sudo apt install -y ${MISSING_DEPS[*]} 2>&1 | grep -E "^(Unable|E:|Package|Setting)" || true
                    ;;
                dnf)
                    sudo dnf install -y ${MISSING_DEPS[*]} 2>&1 | grep -E "^(Error|No package)" || true
                    ;;
                yum)
                    sudo yum install -y ${MISSING_DEPS[*]} 2>&1 | grep -E "^(Error|No package)" || true
                    ;;
                pacman)
                    sudo pacman -S --noconfirm ${MISSING_DEPS[*]} 2>&1 | grep -E "^(error|warning)" || true
                    ;;
            esac
        else
            echo "[WARNING] sudo is not available here (no password entered), system packages were not installed."
        fi
        
        # Check which packages failed to install
        for pkg in "${MISSING_DEPS[@]}"; do
            if ! eval "$PKG_CHECK_CMD '$pkg' > /dev/null 2>&1"; then
                FAILED_DEPS+=("$pkg")
            fi
        done
        
        if [ ${#FAILED_DEPS[@]} -eq 0 ]; then
            echo "[OK] System dependencies installed"
        else
            echo "[WARNING] Some packages could not be installed automatically"
            echo "Failed packages: ${FAILED_DEPS[*]}"
        fi
    else
        echo "[OK] System dependencies already installed"
    fi
    echo ""
fi

# Critical section: require successful completion
set -e

# Install Python dependencies
echo "Installing Python dependencies..."
if [ -f "install/requirements.txt" ]; then
    if ! pip install -r install/requirements.txt -q; then
        echo "[ERROR] Failed to install Python dependencies from install/requirements.txt"
        exit 1
    fi
    echo "[OK] Python dependencies installed"
else
    echo "[ERROR] install/requirements.txt not found!"
    exit 1
fi

# Install CUDA toolkit for the Numba CUDA path if a GPU stack is available.
if ensure_cuda_runtime >/tmp/fidwac_cuda_detect.log 2>&1; then
    CUDA_ROOT="$(detect_cuda_root)"
    export CUDA_HOME="$CUDA_ROOT"
    export PATH="$CUDA_ROOT/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_ROOT/lib:${LD_LIBRARY_PATH:-}"
    echo "[OK] CUDA runtime detected: $CUDA_ROOT"
    if bash "$PROJECT_ROOT/engine/gpu_u8/build.sh" >/tmp/fidwac_gpu_build.log 2>&1; then
        echo "[OK] GPU CUDA libraries built successfully"
    else
        echo "[WARN] GPU CUDA libraries could not be built. CPU-only mode remains active. See /tmp/fidwac_gpu_build.log"
    fi
else
    echo "[WARN] No CUDA toolkit detected; GPU path is disabled, CPU mode remains active."
fi

echo ""

# Allow non-fatal errors again
set +e

# libjpeg-turbo (JPEG quality mode, cm=1): built from install/turbojpeg (optional, non-fatal).
# Online, install/turbojpeg is refreshed to the newest GitHub release; offline, the copy there is used.
JPEG_SRC="$PROJECT_ROOT/install/turbojpeg"
JPEG_PREFIX="$VENV_DIR/libjpeg-turbo"
JPEG_LOG="/tmp/fidwac_libjpeg_build.log"
JPEG_REPO="https://github.com/libjpeg-turbo/libjpeg-turbo"
JPEG_TMP="$(mktemp -d)" || JPEG_TMP=""
if [ -z "$JPEG_TMP" ]; then
    echo "[WARN] No temporary directory, skipping libjpeg-turbo (JPEG quality mode, cm=1 unavailable)."
else
    LATEST_TAG="$(curl -fsSL --connect-timeout 15 -o /dev/null -w '%{url_effective}' "$JPEG_REPO/releases/latest" 2>/dev/null | sed -n 's#.*/releases/tag/\([0-9][0-9.]*\)$#\1#p')"
    LOCAL_VER="$(sed -n 's/^set(VERSION \([0-9.]*\))/\1/p' "$JPEG_SRC/CMakeLists.txt" 2>/dev/null | head -1)"
    if [ -z "$LATEST_TAG" ]; then
        echo "[WARN] GitHub not reachable, using install/turbojpeg as it is (${LOCAL_VER:-missing})."
    elif [ "$LOCAL_VER" != "$LATEST_TAG" ]; then
        echo "Downloading libjpeg-turbo $LATEST_TAG (latest release) ..."
        if curl -fsSL --retry 2 --connect-timeout 15 -o "$JPEG_TMP/src.tar.gz" \
                "$JPEG_REPO/releases/download/$LATEST_TAG/libjpeg-turbo-$LATEST_TAG.tar.gz" \
           && tar -xzf "$JPEG_TMP/src.tar.gz" -C "$JPEG_TMP"; then
            rm -rf "$JPEG_SRC"
            cp -r "$JPEG_TMP/libjpeg-turbo-$LATEST_TAG" "$JPEG_SRC"
            LOCAL_VER="$LATEST_TAG"
        else
            echo "[WARN] Download failed, using install/turbojpeg as it is (${LOCAL_VER:-missing})."
        fi
    fi
    if [ -f "$JPEG_SRC/CMakeLists.txt" ] && command -v cmake > /dev/null 2>&1; then
        echo "Building libjpeg-turbo ${LOCAL_VER:-unknown} into $JPEG_PREFIX ..."
        if cmake -S "$JPEG_SRC" -B "$JPEG_TMP/build" -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_INSTALL_PREFIX="$JPEG_PREFIX" -DCMAKE_INSTALL_LIBDIR=lib \
                -DENABLE_STATIC=OFF -DWITH_TESTS=OFF -DWITH_TOOLS=OFF > "$JPEG_LOG" 2>&1 \
           && cmake --build "$JPEG_TMP/build" -j"$(nproc)" >> "$JPEG_LOG" 2>&1 \
           && cmake --install "$JPEG_TMP/build" >> "$JPEG_LOG" 2>&1 \
           && [ -f "$JPEG_PREFIX/lib/libturbojpeg.so.0" ]; then
            echo "[OK] $(grep -a -m1 -o 'libjpeg-turbo version [0-9.]*' "$JPEG_PREFIX/lib/libturbojpeg.so.0") installed to $JPEG_PREFIX"
            rm -f "$JPEG_LOG"
        else
            echo "[WARN] libjpeg-turbo not built, JPEG quality mode (cm=1) is unavailable. See $JPEG_LOG"
        fi
    else
        echo "[WARN] install/turbojpeg or cmake missing, JPEG quality mode (cm=1) is unavailable."
    fi
    rm -rf "$JPEG_TMP"
fi
echo ""

# SZ3 backend (cm=8): build pysz from install/sz3 (optional, non-fatal).
# Build from a temporary copy so cmake/setuptools output never lands in install/sz3.
if [ -f "install/sz3/tools/pysz/setup.py" ]; then
    echo "Building the SZ3 backend (pysz) from install/sz3 ..."
    PYSZ_LOG="/tmp/fidwac_pysz_build.log"
    SZ3_TMP="$(mktemp -d)"
    cp -r install/sz3 "$SZ3_TMP/sz3"
    if SZ3_SRC="$SZ3_TMP/sz3" pip install --force-reinstall --no-deps \
            "$SZ3_TMP/sz3/tools/pysz" > "$PYSZ_LOG" 2>&1; then
        echo "[OK] SZ3 backend installed from install/sz3"
        rm -f "$PYSZ_LOG"
    else
        echo "[WARN] SZ3 backend not built, --method sz3 is unavailable. See $PYSZ_LOG"
        echo "       Needs cmake >= 3.24, a C++ compiler and zstd development files (libzstd-dev)."
    fi
    rm -rf "$SZ3_TMP"
else
    echo "[WARN] install/sz3 not found, --method sz3 is unavailable."
fi
echo ""

# SZ3 command-line tool: external comparator for analysis/tests/run_tests.py (optional, non-fatal).
# The bundled install/sz3 keeps only the pysz binding (tools/sz3 is absent), so the CLI
# is built from the matching upstream release into install/sz3/bin/sz3 and linked into
# the venv. The test runner checks install/sz3 first, which also resolves its version.
SZ3_CLI_BIN="$PROJECT_ROOT/install/sz3/bin/sz3"
if [ -x "$SZ3_CLI_BIN" ]; then
    echo "[OK] sz3 CLI already present: $SZ3_CLI_BIN"
else
    SZ3_VER="$(sed -n 's/^project(SZ3 VERSION \([0-9.]*\)).*/\1/p' install/sz3/CMakeLists.txt | head -1)"
    if [ -n "$SZ3_VER" ] && command -v cmake > /dev/null 2>&1 && command -v git > /dev/null 2>&1; then
        echo "Building the SZ3 $SZ3_VER command-line tool (external comparator for analysis/tests) ..."
        SZ3C_TMP="$(mktemp -d)"
        if git clone --depth 1 --branch "v$SZ3_VER" https://github.com/szcompressor/SZ3.git "$SZ3C_TMP/SZ3" > /dev/null 2>&1 \
           && cmake -S "$SZ3C_TMP/SZ3" -B "$SZ3C_TMP/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF > /dev/null 2>&1 \
           && cmake --build "$SZ3C_TMP/build" --target sz3 -j"$(nproc)" > /dev/null 2>&1 \
           && [ -x "$SZ3C_TMP/build/tools/sz3/sz3" ]; then
            mkdir -p "$PROJECT_ROOT/install/sz3/bin"
            cp "$SZ3C_TMP/build/tools/sz3/sz3" "$SZ3_CLI_BIN"
            ln -sf "$SZ3_CLI_BIN" "$VENV_DIR/bin/sz3"
            echo "[OK] sz3 $SZ3_VER CLI installed to install/sz3/bin/sz3"
        else
            echo "[WARN] sz3 CLI not built, the 'sz3' test variants are unavailable."
        fi
        rm -rf "$SZ3C_TMP"
    else
        echo "[WARN] sz3 CLI skipped (needs git, cmake and install/sz3/CMakeLists.txt)."
    fi
fi
echo ""

# 7-Zip-zstd (7zz, https://github.com/mcmilk/7-Zip-zstd): archive tool for .7z output (optional, non-fatal).
# Installs the newest Linux release to /usr/local/bin/7zz; sudo asks for the password in this terminal.
echo "Installing 7-Zip-zstd (7zz) ..."
if ! python3 install/install_7zz.py; then
    echo "[WARN] 7zz is not installed: .7z archives fall back to py7zr (slower). See install/README.md."
fi
echo ""

# Verify installation
echo "Verifying installation..."
python3 -c "import numpy, scipy, rasterio, msgpack, py7zr, tqdm, numba, flask; print('[OK] Core dependencies')" 
if [ $? -ne 0 ]; then
    echo "[ERROR] Core dependencies verification failed!"
    exit 1
fi

# TurboJPEG (JPEG quality mode, cm=1) is optional; a failure does not stop the setup.
echo ""
echo "Checking TurboJPEG (JPEG quality mode, cm=1)..."
if python3 -c "from compress.turbojpeg_loader import open_turbojpeg; open_turbojpeg()" > /dev/null 2>&1; then
    echo "[OK] TurboJPEG ready"
else
    echo "[WARN] TurboJPEG not available, JPEG quality mode (cm=1) is disabled."
    echo "       Needs cmake (nasm adds SIMD); rerun this script or see install/README.md."
fi

echo ""
echo "========================================"
echo "INSTALLATION COMPLETED"
echo "========================================"
echo ""

# Display any missing dependencies
if [ ${#FAILED_DEPS[@]} -gt 0 ]; then
    echo "[⚠️  WARNING] Some system dependencies could not be installed automatically:"
    echo ""
    for pkg in "${FAILED_DEPS[@]}"; do
        echo "  • $pkg"
    done
    echo ""
    echo "To install them manually, use your package manager:"
    case "$PKG_MANAGER" in
        apt)
            echo "  sudo apt install ${FAILED_DEPS[*]}"
            ;;
        dnf)
            echo "  sudo dnf install ${FAILED_DEPS[*]}"
            ;;
        yum)
            echo "  sudo yum install ${FAILED_DEPS[*]}"
            ;;
        pacman)
            echo "  sudo pacman -S ${FAILED_DEPS[*]}"
            ;;
        *)
            echo "  Install these packages using your distribution's package manager"
            ;;
    esac
    echo ""
    echo "The application may work without them, but some features might be limited."
    echo ""
fi

echo ""
echo "To run the application:"
echo "  source $VENV_DIR/bin/activate"
echo "  python3 app.py"
echo ""
echo "Or use the script: run_linux.sh or in windows run_windows.bat"
echo "Web GUI address: http://127.0.0.1:5010/"
echo ""
