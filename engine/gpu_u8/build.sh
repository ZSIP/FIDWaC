#!/bin/bash
# Builds the CUDA libraries of the GPU batch path.
# -fmad=false is required: no FMA contraction, so the sums round like numpy's.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_ROOT="${CUDA_HOME:-${NV:-}}"

if [ -z "$CUDA_ROOT" ]; then
  for cand in \
    "/usr/local/cuda" \
    "/opt/cuda" \
    "$HOME/.fidwac/venv/lib/python3.13/site-packages/nvidia/cu13" \
    "$HOME/.fidwac/venv/lib/python3.13/site-packages/nvidia/cu12" \
    "$HOME/.fidwac/venv/lib/python3.13/site-packages/nvidia/cu11"; do
    if [ -x "$cand/bin/nvcc" ]; then
      CUDA_ROOT="$cand"
      break
    fi
  done
fi

if [ -z "$CUDA_ROOT" ] && command -v nvcc >/dev/null 2>&1; then
  CUDA_ROOT="$(dirname "$(dirname "$(command -v nvcc)")")"
fi

if [ -z "$CUDA_ROOT" ] || [ ! -x "$CUDA_ROOT/bin/nvcc" ]; then
  echo "[WARN] CUDA toolkit / nvcc not found; skipping GPU library build. Install nvidia-cuda-nvcc-cu12 or CUDA toolkit and rerun." >&2
  exit 0
fi

NVCC="$CUDA_ROOT/bin/nvcc"
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib:${LD_LIBRARY_PATH:-}"

ARCH="${FIDWAC_CUDA_ARCH:-}"
if [ -z "$ARCH" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 || true)"
    CAP="${CAP//[[:space:]]/}"
    if [ -n "$CAP" ]; then
      ARCH="sm_${CAP//.}"
    fi
  fi
fi
if [ -z "$ARCH" ]; then
  ARCH="sm_75"
fi

echo "[OK] CUDA toolkit detected at $CUDA_ROOT (arch=$ARCH)"

# one library per block edge (8, 16, 32): the kernels are compiled for NN = edge * edge
for bsz in 8 16 32; do
  for src in search_u8_phases:libfidwac_u8_phases rgb_pre:libfidwac_rgb session:libfidwac_session; do
    "$NVCC" -O3 -arch="$ARCH" -fmad=false -DBSZ="$bsz" -shared -Xcompiler -fPIC -cudart shared \
      -I"$CUDA_ROOT/include" -L"$CUDA_ROOT/lib" -L"$CUDA_ROOT/lib64" \
      "$HERE/${src%%:*}.cu" -o "$HERE/${src##*:}_n${bsz}.so"
  done
done

echo "built: $(ls "$HERE"/*.so | xargs -n1 basename | tr '\n' ' ')"
