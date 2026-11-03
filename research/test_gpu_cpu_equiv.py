#!/usr/bin/env python3
"""GPU vs CPU bit-equivalence test for the u8 lockstep search and the cm=6 RGB pre-verify.

The GPU kernels mirror engine/u8_lockstep.py in float64 with the same summation order, but a
different device can still round differently (FMA contraction, libm). The pipeline handles this
by FLAGging rows the GPU cannot decide exactly (IDCT rounding ties, L2 precheck boundary):
search_rows recomputes those rows on the CPU and patches them back into VRAM. This test
verifies the contract end to end on a synthetic cm=6 batch:

  1. search:   search_rows(GpuSession) == search_rows(None)  bitwise (LF, ERR, RES)
  2. rgb:      sess.rgb() == _pre_err_cpu()                   bitwise on non-tie rows

Any mismatch means the GPU arithmetic drifted from the CPU reference and the FLAG/tie safety
net has a hole - the same check a GPU integer-refine kernel would need before it can replace
the CPU loop.

Run:  python3 research/test_gpu_cpu_equiv.py
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from core.dct import dct2, _get_idct_basis, _get_zigzag_indices  # noqa: E402
from engine.blocks import rgb_to_ycbcr, scaled_backscan, _pre_err_cpu  # noqa: E402
from engine.gpu_u8.session import GpuSession  # noqa: E402
from engine.u8_lockstep import search_rows  # noqa: E402

N = 8
NN = N * N
M = 2048          # rows (>= GPU_MIN_ROWS so the session path is taken)
BASE_ACC = 2.0
CH = ("Y", "Cb", "Cr")


def build_batch(seed=42):
    """Synthetic cm=6 batch: smooth, noisy, flat-gray and near-black blocks, like the pipeline."""
    rng = np.random.default_rng(seed)
    r = np.zeros((M, N, N), dtype=np.float32)
    g = np.zeros((M, N, N), dtype=np.float32)
    b = np.zeros((M, N, N), dtype=np.float32)
    kinds = rng.integers(0, 4, M)
    for k in range(M):
        kind = kinds[k]
        if kind == 0:      # smooth gradient
            base = rng.integers(20, 220, 3)
            ramp = np.linspace(0, 1, N * N).reshape(N, N).astype(np.float32)
            r[k] = base[0] + 30 * ramp
            g[k] = base[1] - 20 * ramp
            b[k] = base[2] + 10 * ramp
        elif kind == 1:    # hard noise (many coefficients, ties in rounding)
            amp = rng.integers(10, 80)
            r[k] = 128 + rng.integers(-amp, amp + 1, (N, N))
            g[k] = 128 + rng.integers(-amp, amp + 1, (N, N))
            b[k] = 128 + rng.integers(-amp, amp + 1, (N, N))
        elif kind == 2:    # flat gray with dither (small chroma, like the L=0 refinement case)
            v = rng.integers(100, 160)
            r[k] = v + rng.integers(0, 2, (N, N))
            g[k] = v + rng.integers(0, 2, (N, N))
            b[k] = v + rng.integers(0, 2, (N, N))
        else:              # near-black borders (but not the all-zero marker)
            r[k] = rng.integers(0, 3, (N, N))
            g[k] = rng.integers(0, 3, (N, N))
            b[k] = rng.integers(0, 3, (N, N))
    return np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)


def to_zz_org(plane_f32, dec):
    """plane (M,8,8) float32 -> ZZ (zigzag, float64) + ORG (raster, float64), as compress_channel_rgb."""
    A = plane_f32.astype(np.float64) - 128.0
    dct = np.round(dct2(A, 2), dec)
    zz = np.ascontiguousarray(dct.reshape(M, -1)[:, _get_zigzag_indices(N)].astype(np.float64))
    org = np.ascontiguousarray(A.reshape(M, NN))
    return zz, org


def main():
    r, g, b = build_batch()
    planes = dict(zip(CH, rgb_to_ycbcr(r, g, b)))
    basis2 = np.ascontiguousarray(_get_idct_basis(N, 2).reshape(NN, NN))
    bs = scaled_backscan(10, N)

    u = M
    U = np.arange(u, dtype=np.int64)
    failures = 0

    for sf in (1, 10):
        dec = 0 if sf == 1 else 1
        ZZ = {}
        ORG = {}
        for ch in CH:
            ZZ[ch], ORG[ch] = to_zz_org(planes[ch].astype(np.float32), dec)

        sess = GpuSession([ZZ[c] for c in CH], [ORG[c] for c in CH], basis2, rgb=(r, g, b))
        try:
            for mult in (1.0, 2.5):
                acc = float(mult) * BASE_ACC
                ACC = np.full(u, acc)
                SF = np.full(u, float(sf))
                BS = np.full(u, bs, dtype=np.int64)

                flag_total = 0
                for ci, ch in enumerate(CH):
                    # FLAG stats first: this direct kernel run leaves the flagged rows unfixed in
                    # VRAM, but search_rows below re-runs the kernel and patches them again, so the
                    # session state stays consistent for the rgb stage at the end.
                    _, _, _, fl = sess.search(ci, U, ACC, SF, BS)
                    flag_total += int(fl.sum())
                    LF_c, ERR_c, RES_c, _ = search_rows(None, 0, U, ZZ[ch], ORG[ch],
                                                       ACC, SF, BS, basis2)
                    LF_g, ERR_g, RES_g, used = search_rows(sess, ci, U, ZZ[ch], ORG[ch],
                                                           ACC, SF, BS, basis2)
                    assert used, "GPU session was not used (below GPU_MIN_ROWS?)"

                    ok_lf = np.array_equal(LF_c, LF_g)
                    ok_res = np.array_equal(RES_c, RES_g)
                    err_diff = ERR_c != ERR_g
                    ok_err = not err_diff.any()
                    tag = f"sf={sf} mult={mult} ch={ch}"
                    if not (ok_lf and ok_res and ok_err):
                        failures += 1
                        bad = int(err_diff.sum())
                        dmax = float(np.abs(ERR_c - ERR_g).max()) if bad else 0.0
                        print(f"FAIL {tag}: LF equal={ok_lf}, RES equal={ok_res}, "
                              f"ERR mismatches={bad}/{u}, max|dERR|={dmax:.3e}")
                    else:
                        print(f"ok   {tag}: LF/ERR/RES bitwise identical "
                              f"(flagged-for-CPU rows: {int(fl.sum())}/{u})")

                # RGB pre-verify stage on the same session state (all channels searched and patched)
                err_g, tie_g = sess.rgb(U, float(sf))
                res_by_ch = {}
                for ch in CH:
                    _, _, RES_ch, _ = search_rows(None, 0, U, ZZ[ch], ORG[ch], ACC, SF, BS, basis2)
                    res_by_ch[ch] = RES_ch
                pl = {}
                for ch in CH:
                    cdq = res_by_ch[ch].astype(np.float64) / sf
                    pl[ch] = np.clip((cdq @ basis2).reshape(u, N, N) + 128.0, 0.0, 255.0)
                err_c, tie_c = _pre_err_cpu(pl["Y"], pl["Cb"], pl["Cr"],
                                            r[U], g[U], b[U])
                free = ~tie_g
                same = err_g[free] == err_c[free]
                if same.all():
                    print(f"ok   sf={sf} mult={mult} rgb: err bitwise identical on "
                          f"{int(free.sum())}/{u} non-tie rows (GPU ties: {int(tie_g.sum())})")
                else:
                    failures += 1
                    d = np.abs(err_g[free] - err_c[free])
                    print(f"FAIL sf={sf} mult={mult} rgb: {int((~same).sum())} rows differ, "
                          f"max|dERR|={d.max():.3e}")
        finally:
            sess.close()

    if failures:
        print(f"\n{failures} FAILURES - GPU arithmetic drifted from the CPU reference")
        sys.exit(1)
    print("\nALL OK - GPU path (with FLAG/tie CPU fixups) is bitwise identical to the CPU path")


if __name__ == "__main__":
    main()
