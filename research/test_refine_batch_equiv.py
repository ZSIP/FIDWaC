#!/usr/bin/env python3
"""Bit-equivalence test: _integer_refine_batch vs the per-block _integer_refine_block.

The batch must reproduce, coefficient for coefficient, the trajectory of the per-block loop:
same trial order (channel, coefficient, step), same acceptance rule, same iteration stops, and
the same per-block err_of arithmetic (idct2 along axis 0 first). Blocks that need the pair-move
phase go through the per-block fallback inside the batch, so they match by construction.

Run:  python3 research/test_refine_batch_equiv.py
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from engine.blocks import _integer_refine_batch, _integer_refine_block, _LOCKSTEP_CH

N = 8
NN = 64
SF = 10.0
ACC = 2.0


def make_blocks(rng, kind, count):
    """q_start + originals for one difficulty class."""
    q = np.zeros((count, NN))
    r8 = np.zeros((count, N, N))
    g8 = np.zeros((count, N, N))
    b8 = np.zeros((count, N, N))
    for i in range(count):
        if kind == "easy":        # near the bound, single moves fix it
            q[i] = rng.integers(-3, 4, NN)
            base = 120 + rng.integers(-10, 10, 3)
            noise = rng.integers(-2, 3, (N, N))
            r8[i] = base[0] + noise
            g8[i] = base[1] + noise
            b8[i] = base[2] + noise
        elif kind == "hard":      # well above the bound, many moves needed
            q[i] = rng.integers(-25, 26, NN)
            r8[i] = rng.integers(0, 256, (N, N))
            g8[i] = rng.integers(0, 256, (N, N))
            b8[i] = rng.integers(0, 256, (N, N))
        else:                     # near-flat chroma, hard luma (the L=0 refinement case)
            q[i, :8] = rng.integers(-40, 40, 8)
            r8[i] = rng.integers(0, 256, (N, N))
            g8[i] = r8[i] + rng.integers(-1, 2, (N, N))
            b8[i] = r8[i] + rng.integers(-1, 2, (N, N))
    return {ch: q.copy() for ch in _LOCKSTEP_CH}, r8, g8, b8


def run_case(name, q_start, r8, g8, b8, acc):
    K = q_start["Y"].shape[0]
    q_b = {ch: np.zeros((K, NN)) for ch in _LOCKSTEP_CH}
    e_b = np.zeros(K)
    for j in range(K):
        q1, e1 = _integer_refine_block({ch: q_start[ch][j] for ch in _LOCKSTEP_CH},
                                       r8[j], g8[j], b8[j], SF, acc)
        for ch in _LOCKSTEP_CH:
            q_b[ch][j] = q1[ch]
        e_b[j] = e1
    q_v, e_v = _integer_refine_batch(q_start, r8, g8, b8, SF, acc)
    ok_q = all(np.array_equal(q_b[ch], q_v[ch]) for ch in _LOCKSTEP_CH)
    ok_e = np.array_equal(e_b, e_v)
    solved = int((e_v <= acc).sum())
    print(f"{name:<28} K={K:<4} solved={solved:<4} q identical: {ok_q}   err identical: {ok_e}")
    return ok_q and ok_e


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    ok = True
    ok &= run_case("easy (singles suffice)", *make_blocks(rng, "easy", 40), ACC)
    ok &= run_case("hard (pairs, fallback)", *make_blocks(rng, "hard", 25), ACC)
    ok &= run_case("mixed luma/chroma", *make_blocks(rng, "mixed", 30), ACC)
    ok &= run_case("easy, loose acc=5", *make_blocks(rng, "easy", 20), 5.0)
    print("ALL OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)
