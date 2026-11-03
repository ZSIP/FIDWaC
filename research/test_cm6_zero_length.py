#!/usr/bin/env python3
"""Regression test: cm=6 L == 0 blocks must decode as the neutral 128, not black.

The integer refinement (engine/blocks.py, _integer_refine_block) can zero out every
quantised coefficient of a channel; the encoder then stores L = 0 and its own error
check reconstructs that channel as the neutral plane (0 + 128). Before the fix the
decoder filled L == 0 blocks with 0.0 (a black-block marker inherited from the
single-channel path, where planes are not offset by 128), which painted Cb = 0 as
blue/green shifts of up to 128 DN while the archive still claimed VT.

Run:  python3 research/test_cm6_zero_length.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compress.decompression import _decode_dct_channel_rgb, _reconstruct_plane_batched
from core.codec import dc_median_ac_encode

BLOCK = 8


def _ch_data(lengths_y, lengths_cb, lengths_cr, coeffs_y, coeffs_cb, coeffs_cr, sf=1):
    """Build the flat cm=6 ch_data dict with real encoded streams."""
    bpr = len(lengths_y)
    dc_y, ac_y = dc_median_ac_encode(np.asarray(coeffs_y, dtype=np.int64), list(lengths_y), bpr)
    dc_cb, ac_cb = dc_median_ac_encode(np.asarray(coeffs_cb, dtype=np.int64), list(lengths_cb), bpr)
    dc_cr, ac_cr = dc_median_ac_encode(np.asarray(coeffs_cr, dtype=np.int64), list(lengths_cr), bpr)
    return {
        "l": [list(lengths_y), list(lengths_cb), list(lengths_cr)],
        "c": [dc_y, ac_y, dc_cb, ac_cb, dc_cr, ac_cr],
        "sf": sf,
    }


def test_batched_plane_l0_is_neutral():
    """_reconstruct_plane_batched: L == 0 block reconstructs to 128, L == -1 to -9999."""
    lengths = [1, 0, -1, 1]
    coeffs = [3, 3]
    plane = _reconstruct_plane_batched(coeffs, lengths, BLOCK, 1, BLOCK, 4 * BLOCK)
    assert plane is not None, "batched path refused consistent data"
    b0 = plane[:, 0:8]
    b1 = plane[:, 8:16]      # L == 0 -> neutral
    b2 = plane[:, 16:24]    # L == -1 -> NoData marker
    b3 = plane[:, 24:32]
    assert np.allclose(b1, 128.0), f"L == 0 block is not neutral 128: {b1.min()}..{b1.max()}"
    assert np.all(b2 == -9999.0), "L == -1 block is not the NoData marker"
    assert np.allclose(b0, b3) and not np.allclose(b0, 128.0), "L > 0 blocks lost their coefficients"
    print("ok  batched: L=0 -> 128, L=-1 -> -9999, L>0 -> coefficients")


def test_cm6_cb_l0_decodes_neutral():
    """The reported bug: L(Y)=1, L(Cb)=0, L(Cr)=1 -> RGB must be neutral gray, not shifted."""
    data = _ch_data([1], [0], [1], [0], [], [0])
    rgb = _decode_dct_channel_rgb(data, BLOCK, (BLOCK, BLOCK))
    for ch in "RGB":
        v = rgb[ch]
        assert np.allclose(v, 128.0, atol=1e-3), f"{ch} is not neutral 128: {v.min()}..{v.max()}"
    print("ok  cm=6: Cb with L=0 decodes as 128 (gray), not 0 (color shift)")


def test_cm6_y_l0_decodes_gray_not_black():
    """Y zeroed by the integer refinement: L(Y)=0 -> gray 128, the black marker must not fire."""
    data = _ch_data([0], [1], [1], [], [0], [0])
    rgb = _decode_dct_channel_rgb(data, BLOCK, (BLOCK, BLOCK))
    for ch in "RGB":
        v = rgb[ch]
        assert np.allclose(v, 128.0, atol=1e-3), f"{ch} is not neutral 128: {v.min()}..{v.max()}"
    print("ok  cm=6: Y with L=0 decodes as gray 128, not black")


def test_cm6_nodata_marker_still_black():
    """L == -1 (NoData) keeps the old behaviour: the block is forced black before restoration."""
    data = _ch_data([-1], [-1], [-1], [], [], [])
    rgb = _decode_dct_channel_rgb(data, BLOCK, (BLOCK, BLOCK))
    for ch in "RGB":
        v = rgb[ch]
        assert np.all(v == 0.0), f"{ch}: NoData block is not black: {v.min()}..{v.max()}"
    print("ok  cm=6: L=-1 (NoData) still forces black")


def test_cm6_all_zero_marker_is_black():
    """L == 0 in all three channels is the deliberate all-black (zero RGB) marker: stays black."""
    data = _ch_data([0], [0], [0], [], [], [])
    rgb = _decode_dct_channel_rgb(data, BLOCK, (BLOCK, BLOCK))
    for ch in "RGB":
        v = rgb[ch]
        assert np.all(v == 0.0), f"{ch}: all-zero marker is not black: {v.min()}..{v.max()}"
    print("ok  cm=6: all-three L=0 marker still decodes black")


def test_cm6_l0_next_to_real_coefficients():
    """Two blocks: a coded one and one with chroma zeroed by the refinement - only chroma is neutral."""
    data = _ch_data([1, 1], [1, 0], [1, 0], [40, 0], [0], [0])
    # Y: block0 DC=40 (visible), block1 DC=0 (neutral Y); Cb/Cr: block0 coded, block1 L=0
    rgb = _decode_dct_channel_rgb(data, BLOCK, (BLOCK, 2 * BLOCK))
    for ch in "RGB":
        left, right = rgb[ch][:, :BLOCK], rgb[ch][:, BLOCK:]
        assert np.allclose(right, 128.0, atol=1e-3), f"{ch}: chroma L=0 block not neutral"
        assert not np.allclose(left, right, atol=1e-3), f"{ch}: coded block lost its coefficients"
    print("ok  cm=6: chroma L=0 block neutral next to a coded block")


if __name__ == "__main__":
    test_batched_plane_l0_is_neutral()
    test_cm6_cb_l0_decodes_neutral()
    test_cm6_y_l0_decodes_gray_not_black()
    test_cm6_nodata_marker_still_black()
    test_cm6_all_zero_marker_is_black()
    test_cm6_l0_next_to_real_coefficients()
    print("ALL OK")
