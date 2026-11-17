# Resume notes — FIDWAC batch copy (GPU)

The production tree `/mnt/g/temp/FIDWAC/fidwac_2026_recenzje/FIDWAC_v2` was left untouched during this work.
The work was done in a copy, `/home/infostrateg/fidwac_batch_copy`.

## Environment
- Python: `/home/infostrateg/.fidwac/venv/bin/python3` (numba 0.68, numpy 2.5, scipy).
- CUDA: `NV=/home/infostrateg/.fidwac/venv/lib/python3.13/site-packages/nvidia/cu13`
  - set `LD_LIBRARY_PATH=$NV/lib` before running.
  - build: `$NV/bin/nvcc -O3 -arch=sm_75 -fmad=false -shared -Xcompiler -fPIC -cudart shared -L$NV/lib <file>.cu -o <file>.so`
  - `-fmad=false` is required (rounding identical to numpy's).
- GPU: RTX 2080 Ti, 11 GB VRAM (other processes use it too).

## GPU libraries (`engine/gpu_u8/`)
| file | role | status |
|---|---|---|
| `search_u8_phases.cu` → `libfidwac_u8_phases.so` | phase 1 (pre) and phase 3 (quant), batch from RAM | works, bit-identical |
| `rgb_pre.cu` → `libfidwac_rgb.so` | cm=6: RGB error before verification + rounding-tie flags | works, bit-identical |
| `session.cu` → `libfidwac_session.so` | **session in VRAM**: fused pre+verify+quant, RGB from VRAM, row patching | bit-identical (16/16 configurations) |
| `search_u8.cu` → `libfidwac_u8.so` | old fused kernel | removed |
| `phases.py`, `rgb.py`, `session.py` | ctypes wrappers | — |
| `__init__.py` | `warn_once` only | used |

## CPU/numba code (`engine/u8_lockstep.py`)
- `search_u8(...)`: phase 1 (GPU/numba) → `classic_verify` (scipy, vectorised, exact) → phase 3 (GPU/numba).
- `search_rows(sess, ch, U, ...)`: entry point for cm=5/cm=6 with a session; flagged rows are computed on the CPU and patched into VRAM.
- `search_u8_cpu(...)`: CPU only.
- Earlier `tie_rows`/`_fix_tie_rows` (fallback to production) were removed; scipy verification in phase 2 replaces them.

## Changes in `engine/blocks.py`
- `_classify_blocks`: vectorised classification (first pass of cm=5).
- `_batched_refine`: one GPU session per channel.
- `_rgb_batch_lockstep`: one session per batch (3 channels + RGB in VRAM); `_pre_err_cpu` (CPU fallback); the `near` loop is grouped by L (`tensordot` is bitwise equal to single calls).
- The `near` flag also covers YCbCr rounding ties.

## Changes in `compress/compression_dct.py`
- `_trivial_flags` and `_collect_nontrivial`: vectorised.

## Results (verified)
- Decoding of the copy matches production: 10/10 files (after step 2). Archives differ in bytes (7z metadata), pixels are identical.
- Sequential time, file ~800×800×6 uint8: production 123–228 s, copy ~16–19 s (after step 2).
- 10 files in parallel (before the session version): 28.6 s in total.
- `test_session_parity.py`: ALL OK (16 configurations, 0 mismatching rows). The GPU flags 0–1.9% of the rows for the CPU.
- cm=6 differential test (lockstep vs per-block loop): 0 differences (after step 2). For the session version see below.
- IDCT basis sum vs scipy: all rounding differences (78,701 pixels) lie on ties (|frac−0.5|<1e-9), 0 elsewhere — hence the flagging.

## Pending result at the time of the summary
Script `scratchpad/run10/step3_all.sh` (path: `/tmp/claude-1000/-mnt-g-temp-FIDWAC-fidwac-2026-recenzje-FIDWAC-v2/c92afac9-fa53-48ed-ad96-7d8ab6b0c4fd/scratchpad/run10/`):
1. `cm6_diff_step3.out` — cm=6 differential test with the session. Expected: `differing blocks: 0`.
2. `s3_times.txt` + `s3_*.7z` — 10 files sequentially with the session, decoded and compared with `dec_prod_i`.
3. `par3/wall.txt` — 10 files in parallel (total time).
4. `par3/gpu_samples.txt` — samples of `util,VRAM` every 0.5 s (peak VRAM).
After it finishes: check the decoding of `par3/p_i.7z` against `run10/out/dec_prod_i`.

## Open points and next steps
1. Confirm the results of the pending step (in particular 0 cm=6 differences for the session).
2. Check the decoding of files from the parallel run (only done for the version without the session so far).
3. Measure the peak VRAM with 10 processes (samples are in `gpu_samples.txt`).
4. The float32 path (cm=3) had no batch version at that point. Later it got an array search (`engine/f32_arrays.py`).
5. The remaining per-block Python work in `process_block_batch` (first pass) is the next candidate for vectorisation.
6. The scipy verification (phase 2) stays on the CPU, because numba and the basis sum do not give exact rounding (only on ties, which are flagged).
7. Float16: unchanged (by decision).
8. Clean-up: `libfidwac_u8.so` / `search_u8.cu` removed; README updated; the scripts in `tests_batch/` still have hard-coded paths and need fixing.

## How to resume the tests
- Session parity: `python3 tests_batch/test_session_parity.py <tif> 2500 2`
- RGB parity: `python3 tests_batch/rgb_gpu_vs_cpu.py <tif>`
- cm=6 differential: `python3 tests_batch/cm6_diff.py <tif> 1 2 3` (~2 min)
- E2E of one file: `PYTHONPATH=<copy> python3 compress.py -i <tif> -o out.7z --uint8-accuracy 5 --block-size 8 --rgb-channel-indices 1,2,3 --ycbcr-per-block -p 1 --quiet`
- Decoding: run `compress.py -i out.7z -o dir` from the production directory (only the production decoder code is needed there).

---
# UPDATE (2026-10-30) — array path and multi-file driver

## Correction of the measurements
Earlier absolute times were measured while the machine was loaded by other users' processes (load average ~21).
On a clean machine (load ~1): production file1/2/3 = 55.5 / 89.8 / 65.0 s; copy = 6.0 / 6.3 / 5.6 s (before this update).

## What was added
- `engine/blocks.py`: `search_u8_arrays` (common entry: a VRAM session per channel, or numba), `process_rgb_arrays`
  (cm=6 on arrays: float64−128, two scipy DCTs, round(dec), zigzag, as the production prelude), the core
  `_rgb_lockstep_core(idx_list, ZZ, ORG, r_f, g_f, b_f, ...)`; `_rgb_batch_lockstep` is now a wrapper.
- `compress/compression_dct.py`: `_compress_blocks_arrays` (cm=5 on arrays; gates: uint8, n=8, no NoData,
  no zero masks, no predictor, lockstep on) — no per-block tuple lists; the old path is used when it returns `None`.
  In `compress_channel_rgb` the array path comes before `all_blocks` is built.
- `compress_many.py`: multi-file driver (one initialisation per worker process; `--workers K`, `--list`, `-o`).
  Usage: `python3 compress_many.py --list files.txt -o OUT --workers 8 --uint8-accuracy 5
  --block-size 8 --rgb-channel-indices 1,2,3 --ycbcr-per-block --quiet`

## Results (clean machine, 10 files from multi_uint_6ch, every 1000th file of the sorted list)
| mode | total time | per file | decoding vs production |
|---|---|---|---|
| compress.py, one process per file (file 1) | — | 4.8 s | identical |
| compress_many K=1 | 29.8 s | 2.98 s | 10/10 |
| compress_many K=4 | 12.3 s | 1.23 s | 10/10 |
| compress_many K=8 | 11.0 s | 1.10 s | 10/10 |
Production (clean machine): about 55–90 s per file.

## Finding about the forward DCT (important for the GPU decision)
In cm=5 the channel is float32, so `dct2` computes in float32. A float64 DCT (e.g. on the GPU) rounds differently in
19% of the blocks (dec=0, band 1) up to 47% (dec=2); 34–52% of the blocks have a coefficient within 1e-6 of a tie. Exact
reproduction on the GPU is therefore not possible; flagging would send about half of the blocks back to the CPU.
The scipy DCT stays on the CPU (about 0.1–0.3 s per file).
DECISION PENDING: accept a float64 DCT on the GPU, which changes the archives (the error is still verified against the
original, but the archives are no longer identical to production).

## Open points
1. Batching to the VRAM size: the session allocates the whole channel/batch at once (about 5 KB per block); for images of
   about 90 Mpx (e.g. 9357×9653, about 1.4 million blocks) the data must be split into chunks. An allocation error gives a
   CPU fallback (RuntimeError), but there is no chunking.
2. Several files in one GPU session are not implemented (each worker process has its own session per batch).
3. YCbCr on the GPU: float32 arithmetic can be reproduced exactly (no ties), gain about 0.4 s per file — not done.
4. float32 (cm=3) had no batch path at that point. Float16: unchanged.
5. Clean-up: remove `libfidwac_u8.so` / `search_u8.cu`; update the README; fix the hard-coded paths in `tests_batch/`.
