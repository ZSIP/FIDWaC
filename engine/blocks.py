"""
FIDWAC v2 - block processing
"""

from typing import Tuple, List, Optional, Dict, Any, Union

import math
import numpy as np

from core.dct import dct2, to_zigzag, from_zigzag, idct2, _get_zigzag_indices, _get_idct_basis
from engine.utils import encode_mask_rle, decode_mask_rle, interpolate_zeros, integer_range
from engine.refine import refine_dct_array
from engine import gpu_u8
from engine.gpu_u8 import rgb as gpu_rgb
from engine.gpu_u8 import session as gpu_session

ARRAY_EDGES = (8, 16, 32)  # block edges of the batched array search (GPU kernels are built for these)

_predictor_cache: Dict[tuple, Any] = {}  # per-process predictor cache
from config import Config, load_config
from predictor.predictor_uint8 import load_uint8_L_grid, predict_uint8_L_batch

# Mask types for values the DCT cannot represent (NaN, +inf, -inf); restored exactly on decode.
NON_FINITE_VALUES = {3: np.nan, 4: np.inf, 5: -np.inf}
# Mask type 6: valid pixels that would decode to the NoData value; they step downwards. Used only when
# NoData lies inside the dtype range (e.g. uint16 32767, int16 -9999).
NUDGE_FLIP = 6


def non_finite_masks(mat):
    """(mask type, boolean mask) for the NaN, +inf and -inf pixels of a block."""
    return ((3, np.isnan(mat)), (4, np.isposinf(mat)), (5, np.isneginf(mat)))


def process_block(
    block_data: Tuple[int, np.ndarray, Optional[float]],
    config: Optional[Config] = None,
    predicted_L: Optional[int] = None,
    skip_queue: Optional[list] = None,
) -> Tuple[int, Union[List, np.ndarray], float]:
    """Process single image block and return (idx, compressed_data, max_error)."""
    if config is None:
        config = load_config()

    idx, original_matrix, nodata_value = block_data[:3]
    src_dtype = block_data[3] if len(block_data) > 3 else None
    is_uint8_acc = src_dtype in ("uint8", "int8", "uint16", "int16")
    n = original_matrix.shape[0]
    accuracy = config.compression.accuracy
    dct_type = config.compression.dct_type
    decimal = config.compression.decimal_places
    scaling_factor = config.scaling_factor

    if np.all(original_matrix == 0):
        return idx, 0, 0.0

    if nodata_value is not None and np.all(original_matrix == nodata_value):
        return idx, 1, 0.0

    matrix_for_compression = original_matrix.copy()
    has_special_values = False
    masks = []

    if nodata_value is not None:
        nodata_mask = original_matrix == nodata_value
        if np.any(nodata_mask):
            has_special_values = True
            nodata_rle = encode_mask_rle(nodata_mask)
            masks.append([1] + nodata_rle)  # 1 = typ NoData
            matrix_for_compression[nodata_mask] = 0

    for mask_type, special in non_finite_masks(original_matrix):
        if np.any(special):
            has_special_values = True
            masks.append([mask_type] + encode_mask_rle(special))
            matrix_for_compression[special] = 0

    zero_mask = original_matrix == 0
    if nodata_value is not None:
        zero_mask = zero_mask & ~(original_matrix == nodata_value)

    if np.any(zero_mask):
        has_special_values = True
        zero_rle = encode_mask_rle(zero_mask)
        masks.append([2] + zero_rle)  # 2 = typ zero

    if has_special_values:
        nonzero_mask = matrix_for_compression != 0
        if np.any(nonzero_mask):
            matrix_for_compression = interpolate_zeros(matrix_for_compression, nonzero_mask)

    # uint8/uint16 accuracy mode: centre the data before the DCT (0..255 -> -128..127, uint16 -> -32768..32767)
    if is_uint8_acc:
        matrix_for_compression = matrix_for_compression - integer_range(src_dtype)[0]

    dct_matrix = dct2(matrix_for_compression, dct_type)
    dct_matrix = np.round(dct_matrix, decimal)
    org_dct_zigzag = to_zigzag(dct_matrix)

    compressed_array, max_error, _ = refine_dct_array(
        org_dct_zigzag,
        accuracy,
        matrix_for_compression,
        n,
        dct_type,
        scaling_factor,
        config,
        predicted_L,
        skip_queue=skip_queue,
        src_dtype=src_dtype if is_uint8_acc else None,
    )

    if has_special_values:
        result = masks + [compressed_array.tolist()]
    else:
        result = compressed_array

    if is_uint8_acc:
        flip = nodata_flip_mask(
            result, original_matrix, n, scaling_factor,
            nodata_value if config._source_has_nodata else None, src_dtype,
        )
        if flip is not None:
            result = _with_flip_mask(result, flip)
    if isinstance(result, np.ndarray):
        result = result.tolist()

    return idx, result, max_error


def predict_L_arrays(ZZ, config, accuracy, is_ycbcr=False):
    """Predicted L per row (m rows of 64 zigzag coefficients) from Uint8Predictor (lookup grid of the
    scaling factor and accuracy). For cm=6 pass the Y, Cb, Cr rows stacked in that order with
    is_ycbcr=True, so the predictor uses its per-channel grids."""
    from predictor.predictor_uint8 import Uint8Predictor

    predictor = Uint8Predictor(accuracy=float(accuracy), block_size=8, models_dir=config.models_dir,
                               config=config, is_ycbcr=is_ycbcr)
    preds, _ = predictor.predict_from_dct_batch(ZZ)
    return np.asarray(preds, dtype=np.int64)


def search_u8_arrays(ZZ, OR, ACC, SF, BS, LP=None, scales=None, brk=3):
    """Channel search on whole arrays (m, n*n). With a GPU the batch is copied to VRAM once and the
    fused kernel runs there; otherwise numba (prange) and scipy are used. Both give the same results.
    With LP (predicted L per row) and scales the predicted-L acceptance is used (CPU only); brk is
    backscan_break_after. Returns LF (m,), ERR (m,), RES (m, n*n); n = 8, 16 or 32."""
    from engine.u8_lockstep import search_rows, search_rows_pred

    m, nn = ZZ.shape
    n = int(round(nn ** 0.5))
    basis2 = np.ascontiguousarray(_get_idct_basis(n, 2).reshape(nn, nn))
    if LP is not None:
        LF, ERR, RES = search_rows_pred(ZZ, OR, np.arange(m, dtype=np.int64), ACC, SF, BS, basis2,
                                        LP, scales, brk=brk)
        return LF, ERR, RES
    sess = None
    if m >= GPU_MIN_ROWS and gpu_session.available(n):
        try:
            sess = gpu_session.GpuSession([ZZ], [OR], basis2)
        except RuntimeError as exc:
            gpu_u8.warn_once(f"GPU session failed, using CPU: {exc}")
    try:
        LF, ERR, RES, _ = search_rows(sess, 0, np.arange(m, dtype=np.int64), ZZ, OR, ACC, SF, BS,
                                      basis2, brk)
    finally:
        if sess is not None:
            sess.close()
    return LF, ERR, RES


def scaled_backscan(base, n):
    """minimize_backscan for block edge n: the reference search (refine_dct_array) scales it with the
    block size, (n / 8) * sqrt(n / 8)."""
    return int(base * ((n / 8) * math.sqrt(n / 8)))


def _same_array_edge(blocks, get):
    """True when every block has the same edge n and n has array kernels (8, 16, 32)."""
    if not blocks:
        return False
    n = get(blocks[0]).shape[0]
    return n in ARRAY_EDGES and all(get(bd).shape[0] == n for bd in blocks)


def _batched_refine(rec, config, src_dtype):
    """Batched refine_dct_array for the entries collected in the first pass of process_block_batch.
    Blocks with n=8, 16 or 32, dct type 2 and no predicted_L/skip_queue are searched in the array search
    (GPU or numba); the others use refine_dct_array."""
    from engine.u8_lockstep import search_rows

    out = {}
    fast = []
    n0 = rec[0][4] if rec else 8  # one block size per batch
    for entry in rec:
        idx, org, acc, mat, n_, dct_tp, sf_, lp, sq = entry
        if n_ == n0 and n_ in ARRAY_EDGES and dct_tp == 2 and lp is None and not sq and src_dtype == "uint8":
            fast.append(entry)
        else:
            out[idx] = refine_dct_array(
                org, acc, mat, n_, dct_tp, sf_, config, lp, skip_queue=sq, src_dtype=src_dtype
            )
    if not fast:
        return out

    m = len(fast)
    nn = n0 * n0
    ZZ = np.ascontiguousarray(np.stack([np.asarray(e[1], dtype=np.float64) for e in fast])).reshape(m, nn)
    OR = np.ascontiguousarray(
        np.stack([np.asarray(e[3], dtype=np.float64) for e in fast])
    ).reshape(m, nn)
    ACC = np.array([float(e[2]) for e in fast])
    SF = np.array([float(e[6]) for e in fast])
    BS = np.full(m, scaled_backscan(config.model.minimize_backscan, n0), dtype=np.int64)
    LF, ERR, RES = search_u8_arrays(ZZ, OR, ACC, SF, BS)
    for i, e in enumerate(fast):
        L = int(LF[i])
        out[e[0]] = (RES[i, :L].astype(int), float(ERR[i]), None)
    return out


def _classify_blocks(blocks, is_uint8, config):
    """Vectorised block flags of the first pass of process_block_batch, with the same tests as the
    per-block code: 'skip' (all zero or all NoData) and 'special' (NoData or non-finite values inside,
    or zeros with uint8_zero_mask). Returns None when the batch cannot be stacked (mixed shapes or
    NoData values)."""
    if not blocks:
        return None
    nd0 = blocks[0][2] if len(blocks[0]) > 2 else None
    shape0 = np.shape(blocks[0][1])
    for bt in blocks:
        if len(bt) < 3 or np.shape(bt[1]) != shape0:
            return None
        if not (bt[2] is nd0 or (nd0 is not None and bt[2] == nd0) or (nd0 is None and bt[2] is None)):
            return None
    A = np.stack([np.asarray(bt[1]) for bt in blocks])
    ax = tuple(range(1, A.ndim))
    zero = A == 0
    allzero = zero.all(axis=ax)
    if nd0 is not None:
        ndm = A == nd0
        nd_all = ndm.all(axis=ax)
    else:
        ndm = np.zeros(A.shape, dtype=bool)
        nd_all = np.zeros(A.shape[0], dtype=bool)
    unusable = ndm | ~np.isfinite(A)  # NaN / inf are masked like NoData
    zero_masked = zero.any(axis=ax) & (not is_uint8 or config.compression.uint8_zero_mask)
    return {"skip": allzero | nd_all, "special": unusable.any(axis=ax) | zero_masked}


def process_block_batch(
    batch_data: Tuple[int, List], config: Optional[Config] = None,
    _refine_results: Optional[dict] = None, _refine_record: Optional[list] = None,
) -> List[Tuple[int, Any, float]]:
    """Process a batch of blocks with optional L prediction. Returns list of (idx, data, error)."""
    if config is None:
        config = load_config()

    _batch_idx, blocks = batch_data
    # the source dtype is the 4th element of each block tuple
    _src_dtype = blocks[0][3] if blocks and len(blocks[0]) > 3 else None
    _is_uint8_acc = _src_dtype in ("uint8", "int8", "uint16", "int16")

    if (
        _refine_results is None
        and _refine_record is None
        and _is_uint8_acc
        and _array_path_enabled(config)
        and _same_array_edge(blocks, lambda bd: bd[1])
    ):
        rec: list = []
        try:
            process_block_batch(batch_data, config, _refine_record=rec)
        except Exception:  # noqa: BLE001  (atrapowe wyniki w przebiegu zbierajacym)
            return process_block_batch(batch_data, config, _refine_results={})
        return process_block_batch(
            batch_data, config, _refine_results=_batched_refine(rec, config, _src_dtype)
        )
    _center = integer_range(_src_dtype)[0] if _is_uint8_acc else 0.0
    L_predictions = {}
    skip_queues = {}
    results = []

    def _refine_for_block(idx, org, acc, mat, n_, dct_tp, sf_, cfg, lp, sq):
        if _refine_record is not None:
            _refine_record.append((idx, org, acc, mat, n_, dct_tp, sf_, lp, sq))
            return np.zeros(1, dtype=int), 0.0, None
        if _refine_results is not None and idx in _refine_results:
            return _refine_results[idx]
        return refine_dct_array(org, acc, mat, n_, dct_tp, sf_, cfg, lp, skip_queue=sq,
                                src_dtype=_src_dtype)

    decimal = config.compression.decimal_places

    dct_candidates = []  # clean blocks
    special_candidates = []  # blocks with nodata or zeros — masks restore them exactly
    _cls = _classify_blocks(blocks, _is_uint8_acc, config)
    for i, block_tuple in enumerate(blocks):
        try:
            _idx, _mat, _nodata = block_tuple[:3]
        except (TypeError, ValueError):
            continue
        if _cls is None:
            if np.all(_mat == 0):
                continue
            if _nodata is not None and np.all(_mat == _nodata):
                continue
        elif _cls["skip"][i]:
            continue
        elif not _cls["special"][i]:
            dct_candidates.append((i, _idx, _mat - _center if _is_uint8_acc else _mat))
            continue
        _nodata_mask = (
            _mat == _nodata if _nodata is not None else np.zeros(_mat.shape, dtype=bool)
        )
        # Zeros are data, like NoData: a block that mixes zeros with other values gets a zero mask so the
        # zeros decode exactly (same as process_block).
        _unusable = _nodata_mask | ~np.isfinite(_mat)  # NaN and inf are masked like NoData
        _zero_masked = np.any(_mat == 0) and (
            not _is_uint8_acc or config.compression.uint8_zero_mask
        )
        if np.any(_unusable) or _zero_masked:
            _mat_interp = _mat.copy()
            _mat_interp[_unusable] = 0
            nonzero = _mat_interp != 0
            if np.any(nonzero):
                _mat_interp = interpolate_zeros(_mat_interp, nonzero)
            special_candidates.append(
                (
                    i,
                    _idx,
                    _mat_interp - _center if _is_uint8_acc else _mat_interp,
                    _mat,
                    _nodata,
                )
            )
            continue

        dct_candidates.append((i, _idx, _mat - _center if _is_uint8_acc else _mat))

    precomp_map = {}
    special_map = {}  # _idx -> (original_mat, nodata_val)
    all_for_dct = list(dct_candidates)
    if special_candidates:
        for i, _idx, _mat_interp, _mat_orig, _nodata in special_candidates:
            all_for_dct.append((i, _idx, _mat_interp))
            special_map[_idx] = (_mat_orig, _nodata)

    precomp_vectors = None
    if all_for_dct:
        dct_batch = np.array([m for (_, _, m) in all_for_dct])
        dct_batch = dct2(dct_batch, config.compression.dct_type)
        dct_batch = np.round(dct_batch, decimal)

        B = dct_batch.shape[0]
        Nloc = dct_batch.shape[1]
        idx_z = _get_zigzag_indices(Nloc)
        all_vectors = dct_batch.reshape(B, -1)[:, idx_z]

        for k, (i, _idx, _m) in enumerate(all_for_dct):
            precomp_map[_idx] = all_vectors[k]

        precomp_vectors = all_vectors[: len(dct_candidates)]

    # L prediction
    backend = config.model.backend.lower()
    use_predictor = backend in ("heuristic",) or config.model.advanced_heuristic

    if use_predictor:
        try:
            from predictor.predictor import get_predictor, AdvancedHeuristicPredictor
            from predictor.predictor_uint8 import Uint8Predictor

            _sf_override = getattr(config, "_scaling_factor_override", None)
            _sf_cfg = getattr(config.compression, "uint8_scaling_factor", 1)
            if _sf_override is not None:
                _effective_sf = int(_sf_override)
            elif isinstance(_sf_cfg, (list, tuple)):
                _effective_sf = int(_sf_cfg[0]) if _sf_cfg else 1
            else:
                _effective_sf = int(_sf_cfg)
            _uint8_l_scales = tuple(
                float(x) for x in getattr(config.model, "uint8_L_prediction_scales", [1.5])
            )

            _cache_key = (
                backend,
                config.model.advanced_heuristic,
                config.compression.accuracy,
                config.compression.block_size,
                _src_dtype,  # Include src_dtype for uint8 vs float distinction
                _effective_sf,
                _uint8_l_scales,
            )
            if _cache_key not in _predictor_cache:
                _predictor_cache[_cache_key] = get_predictor(
                    backend=backend,
                    advanced_heuristic=config.model.advanced_heuristic,
                    accuracy=config.compression.accuracy,
                    block_size=config.compression.block_size,
                    config=config,
                    src_dtype=_src_dtype,  # Pass src_dtype for uint8 predictor selection
                )
            predictor = _predictor_cache[_cache_key]

            # batch prediction for AdvancedHeuristicPredictor and Uint8Predictor
            if (
                isinstance(predictor, (AdvancedHeuristicPredictor, Uint8Predictor))
                and dct_candidates
            ):
                indices_for_pred = [_idx for (_, _idx, _) in dct_candidates]
                preds, queues = predictor.predict_from_dct_batch(precomp_vectors)
                for k, _idx in enumerate(indices_for_pred):
                    if k < len(preds):
                        L_predictions[_idx] = int(preds[k])
                        if queues and k < len(queues) and queues[k]:
                            skip_queues[_idx] = queues[k]
            else:
                # HeuristicPredictor: per-block prediction
                eligible_blocks = [_mat for (_, _, _mat) in dct_candidates]
                eligible_indices = [_idx for (_, _idx, _) in dct_candidates]

                if eligible_blocks:
                    res = predictor.predict_dct_length_batch(eligible_blocks)
                    if isinstance(res, tuple):
                        preds, queues = res
                    else:
                        preds, queues = res, [[]] * len(res)
                    for k, idx in enumerate(eligible_indices):
                        if k < len(preds):
                            L_predictions[idx] = int(preds[k])
                            if queues and k < len(queues) and queues[k]:
                                skip_queues[idx] = queues[k]
        except (ImportError, RuntimeError):
            pass

    # uint8 L prediction from the lookup grid (ac_abs_mean, zero_ratio); off unless
    # config.model.uint8_use_L_prediction is set.
    if (
        _is_uint8_acc
        and config.model.uint8_use_L_prediction
        and precomp_vectors is not None
        and len(dct_candidates) > 0
    ):
        sf_override = getattr(config, "_scaling_factor_override", None)
        sf_cfg = getattr(config.compression, "uint8_scaling_factor", 1)
        if sf_override is not None:
            sf = int(sf_override)
        elif isinstance(sf_cfg, (list, tuple)):
            sf = int(sf_cfg[0]) if sf_cfg else 1
        else:
            sf = int(sf_cfg)
        cache_key = ("uint8_L_grid", Nloc, sf)
        if cache_key not in _predictor_cache:
            models_dir = config.models_dir if hasattr(config, 'models_dir') else "models"
            _predictor_cache[cache_key] = load_uint8_L_grid(Nloc, sf, models_dir)
        l_grid = _predictor_cache[cache_key]

        u8_indices = [_idx for (_, _idx, _) in dct_candidates]
        n_u8 = min(len(u8_indices), len(precomp_vectors))

        if n_u8 > 0:
            ac_abs_means = np.empty(n_u8, dtype=np.float32)
            zero_ratios = np.empty(n_u8, dtype=np.float32)
            for k in range(n_u8):
                zigzag = precomp_vectors[k]
                ac = zigzag[1:]  # skip DC
                ac_abs_means[k] = float(np.mean(np.abs(ac)))
                zero_ratios[k] = float(np.sum(ac == 0)) / max(1, len(ac))

            default_L = max(1, (Nloc * Nloc) // 2)
            L_preds = predict_uint8_L_batch(ac_abs_means, zero_ratios, l_grid, default_L)

            for k, _idx in enumerate(u8_indices[:n_u8]):
                if _idx not in L_predictions:
                    L_predictions[_idx] = int(L_preds[k])

    for block_tuple in blocks:
        try:
            idx = block_tuple[0]
            L_pred = L_predictions.get(idx)

            # reuse the zigzag computed in the batch pass when available
            if idx in precomp_map:
                _idx, _mat, _nodata = block_tuple[:3]
                # all-zero and all-NoData blocks are skipped in the first pass and never get here

                org_dct_zigzag = precomp_map[idx]

                if idx in special_map:
                    _mat_orig, _nodata_val = special_map[idx]
                    nodata_mask = (
                        _mat_orig == _nodata_val
                        if _nodata_val is not None
                        else np.zeros(_mat_orig.shape, dtype=bool)
                    )
                    zero_mask = (_mat_orig == 0) & ~nodata_mask
                    masks = []
                    if np.any(nodata_mask):
                        masks.append([1] + encode_mask_rle(nodata_mask))
                    if np.any(zero_mask):
                        masks.append([2] + encode_mask_rle(zero_mask))
                    _mat_for_refine = _mat_orig.copy()
                    _mat_for_refine[nodata_mask] = 0
                    for mask_type, special in non_finite_masks(_mat_orig):
                        if np.any(special):
                            masks.append([mask_type] + encode_mask_rle(special))
                            _mat_for_refine[special] = 0
                    nonzero = _mat_for_refine != 0
                    if np.any(nonzero):
                        _mat_for_refine = interpolate_zeros(_mat_for_refine, nonzero)
                    if _is_uint8_acc:
                        _mat_for_refine = _mat_for_refine - _center
                    compressed_array, max_error, _ = _refine_for_block(
                        idx,
                        org_dct_zigzag,
                        config.compression.accuracy,
                        _mat_for_refine,
                        _mat_for_refine.shape[0],
                        config.compression.dct_type,
                        config.scaling_factor,
                        config,
                        L_pred,
                        skip_queues.get(idx),
                    )
                    result = masks + [compressed_array.tolist()]
                    if _is_uint8_acc:
                        flip = nodata_flip_mask(
                            result, _mat_orig, _mat_orig.shape[0], config.scaling_factor,
                            _nodata_val if config._source_has_nodata else None, _src_dtype,
                        )
                        if flip is not None:
                            result = _with_flip_mask(result, flip)
                    results.append((_idx, result, max_error))
                else:
                    _mat_r = _mat - _center if _is_uint8_acc else _mat
                    compressed_array, max_error, _ = _refine_for_block(
                        idx,
                        org_dct_zigzag,
                        config.compression.accuracy,
                        _mat_r,
                        _mat_r.shape[0],
                        config.compression.dct_type,
                        config.scaling_factor,
                        config,
                        L_pred,
                        skip_queues.get(idx),
                    )
                    result = compressed_array
                    if _is_uint8_acc:
                        flip = nodata_flip_mask(
                            result, _mat, _mat.shape[0], config.scaling_factor,
                            _nodata if config._source_has_nodata else None, _src_dtype,
                        )
                        if flip is not None:
                            result = _with_flip_mask(result, flip)
                    results.append((_idx, result, max_error))
            else:
                skip_q = skip_queues.get(idx)
                result = process_block(block_tuple, config, L_pred, skip_queue=skip_q)
                results.append(result)
        except (TypeError, ValueError) as e:
            idx = block_tuple[0] if isinstance(block_tuple, tuple) else 0
            results.append((idx, {"type": "error", "msg": str(e)}, 1.0))

    return results


def reconstruct_block(
    compressed_data: Union[List, np.ndarray],
    n: int,
    scaling_factor: int,
    nodata_value: float = -9999,
    uint8_mode: bool = False,
    src_dtype: Optional[str] = None,
    nodata_guard: Optional[float] = None,
) -> np.ndarray:
    """Reconstruct a block from compressed DCT data.

    The coefficients are the DCT values multiplied by scaling_factor.
    uint8_mode: True for cm=5 (integer accuracy path): add the centre, round and clip to the range of
    src_dtype (uint8 when src_dtype is None: 128, [0, 255]).
    nodata_guard: the source NoData value (None when there is none). In uint8_mode a valid pixel that
    decodes to it is moved one step away, so only masked pixels carry the NoData value.
    """
    if compressed_data == 0:
        return np.zeros((n, n), dtype=np.float32)
    if compressed_data == 1:
        return np.full((n, n), nodata_value, dtype=np.float32)

    has_masks = False
    masks = []
    dct_coeffs = compressed_data

    if isinstance(compressed_data, list) and len(compressed_data) > 1:
        if isinstance(compressed_data[0], list):
            has_masks = True
            dct_coeffs = compressed_data[-1]
            masks = compressed_data[:-1]

    total_coeffs = n * n
    dct_zigzag = np.zeros(total_coeffs, dtype=np.float64)

    if isinstance(dct_coeffs, (list, np.ndarray)):
        L = min(len(dct_coeffs), total_coeffs)
        dct_zigzag[:L] = np.asarray(dct_coeffs[:L], dtype=np.float64) / scaling_factor

    dct_matrix = from_zigzag(dct_zigzag, n)
    reconstructed = idct2(dct_matrix).astype(np.float64)

    if uint8_mode:
        _c, _lo, _hi = integer_range(src_dtype or "uint8")
        reconstructed = np.clip(np.round(reconstructed + _c), _lo, _hi).astype(np.float32)
        if nodata_guard is not None and _lo <= nodata_guard <= _hi:
            # A valid pixel must not decode to the NoData value (it would be read as NoData): step it one
            # value into the data range, down at the top of the range and up otherwise. Mask type 6
            # (written only for NoData inside the range) marks pixels that step the other way.
            hit = reconstructed == nodata_guard
            if np.any(hit):
                step = -1.0 if nodata_guard >= _hi else 1.0
                reconstructed[hit] = nodata_guard + step
                for mask_data in masks:
                    if isinstance(mask_data, list) and len(mask_data) > 1 and mask_data[0] == NUDGE_FLIP:
                        flip = decode_mask_rle(mask_data[1:], (n, n)) & hit
                        reconstructed[flip] = nodata_guard - step

    if has_masks:
        for mask_data in masks:
            if isinstance(mask_data, list) and len(mask_data) > 1:
                mask_type = mask_data[0]
                mask_rle = mask_data[1:]

                if mask_type == 1:  # NoData
                    nodata_mask = decode_mask_rle(mask_rle, (n, n))
                    reconstructed[nodata_mask] = nodata_value
                elif mask_type == 2:  # Zero
                    zero_mask = decode_mask_rle(mask_rle, (n, n))
                    reconstructed[zero_mask] = 0
                elif mask_type in NON_FINITE_VALUES:  # 3 = NaN, 4 = +inf, 5 = -inf
                    special = decode_mask_rle(mask_rle, (n, n))
                    reconstructed[special] = NON_FINITE_VALUES[mask_type]

    return reconstructed


def nodata_flip_mask(
    block_data: Union[List, np.ndarray],
    original: np.ndarray,
    n: int,
    scaling_factor: int,
    nodata_value: Optional[float],
    src_dtype: Optional[str],
) -> Optional[np.ndarray]:
    """Pixels the decoder must step DOWN from the NoData value (mask type 6), or None.

    Only NoData strictly inside the dtype range needs it: the decoder's default step (+1, or -1 at the
    top of the range) moves away from an edge NoData towards every valid original value. The block is
    decoded with reconstruct_block, so the result matches the decoder exactly."""
    rng = integer_range(src_dtype)
    if rng is None or nodata_value is None or not (rng[1] < nodata_value < rng[2]):
        return None
    if isinstance(block_data, np.ndarray):
        block_data = block_data.tolist()
    dec = reconstruct_block(
        block_data, n, scaling_factor, nodata_value, uint8_mode=True, src_dtype=src_dtype
    )
    flip = (dec == nodata_value) & (original < nodata_value)
    return flip if np.any(flip) else None


def _with_flip_mask(result, flip):
    """Add a type-6 mask to a block result (coefficient array, or masks + [coefficients])."""
    mask = [NUDGE_FLIP] + encode_mask_rle(flip)
    if isinstance(result, np.ndarray):
        return [mask, result.tolist()]
    return result[:-1] + [mask, result[-1]]

__all__ = [
    "process_block",
    "process_block_batch",
    "process_block_batch_rgb",
    "process_strip_batch",
    "reconstruct_block",
    "rgb_to_ycbcr",
    "ycbcr_to_rgb",
]


# ---------------------------------------------------------------------------
# RGB <-> YCbCr conversion helpers (per-block)
# ---------------------------------------------------------------------------


def rgb_to_ycbcr(
    R: np.ndarray, G: np.ndarray, B: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an RGB block to YCbCr (JPEG coefficients).

    Standard JPEG conversion:
    Y  = 0.299*R + 0.587*G + 0.114*B
    Cb = -0.169*R - 0.331*G + 0.500*B + 128
    Cr = 0.500*R - 0.419*G - 0.081*B + 128
    """
    Y = np.clip(0.299 * R + 0.587 * G + 0.114 * B, 0, 255)
    Cb = np.clip(-0.169 * R - 0.331 * G + 0.500 * B + 128, 0, 255)
    Cr = np.clip(0.500 * R - 0.419 * G - 0.081 * B + 128, 0, 255)
    return Y, Cb, Cr


def ycbcr_to_rgb(
    Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a YCbCr block back to RGB (inverse JPEG conversion).

    Inverse:
    R = Y + 1.403*(Cr - 128)
    G = Y - 0.344*(Cb - 128) - 0.714*(Cr - 128)
    B = Y + 1.773*(Cb - 128)
    """
    R = np.clip(np.round(Y + 1.403 * (Cr - 128.0)), 0, 255)
    G = np.clip(np.round(Y - 0.344 * (Cb - 128.0) - 0.714 * (Cr - 128.0)), 0, 255)
    B = np.clip(np.round(Y + 1.773 * (Cb - 128.0)), 0, 255)
    return R, G, B


# ---------------------------------------------------------------------------
# Per-block RGB processing with YCbCr validity check
# ---------------------------------------------------------------------------


def process_strip_batch(
    strip_data: Tuple[np.ndarray, int, int, Optional[float]],
    config: Config,
) -> List[Tuple[int, Any, float]]:
    """Compress a strip with the standard refine path (binary search), for elevation data."""
    strip_array, y_start, blocks_per_row, nodata_value = strip_data
    n = config.compression.block_size

    sH, sW = strip_array.shape
    rows_of_blocks = sH // n
    cols_of_blocks = sW // n

    blocks_3d = (
        strip_array.reshape(rows_of_blocks, n, cols_of_blocks, n)
        .transpose(0, 2, 1, 3)
        .reshape(-1, n, n)
    )

    global_row_offset = y_start // n

    # block tuples for process_block_batch
    blocks = []
    for local_idx in range(blocks_3d.shape[0]):
        local_row = local_idx // cols_of_blocks
        local_col = local_idx % cols_of_blocks
        global_idx = (global_row_offset + local_row) * blocks_per_row + local_col
        blocks.append((global_idx, blocks_3d[local_idx], nodata_value))

    # delegate to the batch function
    return process_block_batch((0, blocks), config)


# ---------------------------------------------------------------------------
# Per-block RGB / YCbCr compression (cm=6)
# ---------------------------------------------------------------------------


def _rgb_error_via_decoder(ch_data, n, sf, r_f, g_f, b_f):
    """Maximum RGB error of a block as the decoder reconstructs it: reconstruct_block (scipy idct2),
    clip(+128) and ycbcr_to_rgb. Called only for the candidate that is finally accepted."""
    planes = {}
    for ch in ("Y", "Cb", "Cr"):
        if ch_data[ch]["L"] > 0:
            rec = reconstruct_block(ch_data[ch]["coeffs"].tolist(), n, sf, -9999, uint8_mode=False)
            planes[ch] = np.clip(np.asarray(rec, dtype=np.float64) + 128.0, 0.0, 255.0)
        else:
            planes[ch] = np.full((n, n), 128.0)  # Same as the decoder: no coefficients = neutral
    R_dec, G_dec, B_dec = ycbcr_to_rgb(planes["Y"], planes["Cb"], planes["Cr"])
    if all(ch_data[ch]["L"] == 0 for ch in ("Y", "Cb", "Cr")):
        # all-zero marker: the decoder writes the block black
        R_dec = G_dec = B_dec = np.zeros((n, n), dtype=np.float64)
    _err = float(
        max(np.max(np.abs(r_f - R_dec)), np.max(np.abs(g_f - G_dec)), np.max(np.abs(b_f - B_dec)))
    )
    return _err


_LOCKSTEP_CH = ("Y", "Cb", "Cr")


GPU_MIN_ROWS = 256  # batches below this stay on the CPU (kernel launch + copies)


def _array_path_enabled(config) -> bool:
    """Array search for uint8 (cm=5/cm=6) and float32 channels. Requires the optimisation flags that the
    array search reproduces; the per-block path handles the other cases (heuristic backends, masks)."""
    perf = config.performance
    if not (perf.fast_eval_basis and perf.incremental_backscan and perf.verify_with_full_idct):
        return False
    return True  # backscan and its break count are parameters of the array search (any value)


def _rgb_decoder_err_batch(res_by_ch, lf_by_ch, sf, r_f, g_f, b_f):
    """Batch version of _rgb_error_via_decoder. The transform order is that of reconstruct_block
    (idct2 in 2-D: axis 0 of the block first, then axis 1)."""
    from scipy.fft import idct as _idct
    from core.dct import _get_zigzag_indices

    nn = res_by_ch[_LOCKSTEP_CH[0]].shape[1]
    n = int(round(nn ** 0.5))
    zz_idx = _get_zigzag_indices(n)
    planes = {}
    for ch in _LOCKSTEP_CH:
        rs = res_by_ch[ch]
        u = rs.shape[0]
        flat = np.zeros((u, nn), dtype=np.float64)
        flat[:, zz_idx] = rs.astype(np.float64) / sf
        rec = _idct(_idct(flat.reshape(u, n, n), axis=1, norm="ortho", type=2),
                    axis=2, norm="ortho", type=2)
        # L == 0 leaves the row at zero coefficients, i.e. the neutral plane 128, as in the decoder
        planes[ch] = np.clip(rec + 128.0, 0.0, 255.0)
    R_dec, G_dec, B_dec = ycbcr_to_rgb(planes["Y"], planes["Cb"], planes["Cr"])
    marker = (lf_by_ch["Y"] == 0) & (lf_by_ch["Cb"] == 0) & (lf_by_ch["Cr"] == 0)
    if np.any(marker):
        # all-zero marker: the decoder writes the block black
        R_dec, G_dec, B_dec = R_dec.copy(), G_dec.copy(), B_dec.copy()
        R_dec[marker] = G_dec[marker] = B_dec[marker] = 0.0
    return np.maximum(
        np.maximum(
            np.max(np.abs(r_f - R_dec), axis=(1, 2)),
            np.max(np.abs(g_f - G_dec), axis=(1, 2)),
        ),
        np.max(np.abs(b_f - B_dec), axis=(1, 2)),
    )


def _pre_err_cpu(pY, pCb, pCr, rf, gf, bf):
    """Pre-verification RGB error from the planes, with the arithmetic of ycbcr_to_rgb, and a rounding
    tie flag per row (a pixel whose value before rounding lies within 1e-9 of a half-integer)."""
    xr = pY + 1.403 * (pCr - 128.0)
    xg = pY - 0.344 * (pCb - 128.0) - 0.714 * (pCr - 128.0)
    xb = pY + 1.773 * (pCb - 128.0)
    R = np.clip(np.round(xr), 0, 255)
    G = np.clip(np.round(xg), 0, 255)
    B = np.clip(np.round(xb), 0, 255)
    err = np.maximum(
        np.maximum(
            np.max(np.abs(rf - R), axis=(1, 2)),
            np.max(np.abs(gf - G), axis=(1, 2)),
        ),
        np.max(np.abs(bf - B), axis=(1, 2)),
    )
    tie = np.zeros(pY.shape[0], dtype=bool)
    for x in (xr, xg, xb):
        tie |= np.any(np.abs((x - np.floor(x)) - 0.5) < 1e-9, axis=(1, 2))
    return err, tie


def _rgb_batch_lockstep(blocks, zz_all, cen_all, fallback_multipliers, base_accuracy, sf, dct_tp,
                        config):
    """Multiplier loop in lockstep over all unresolved blocks at once, from per-block lists.
    Returns [(idx, metadata, err)] or None (the caller then uses the per-block loop).
    The array version is _rgb_lockstep_core."""
    m = len(blocks)
    ZZ = {ch: np.ascontiguousarray(np.stack(zz_all[ch]), dtype=np.float64).reshape(m, -1)
          for ch in _LOCKSTEP_CH}
    ORG = {ch: np.ascontiguousarray(np.stack(cen_all[ch]), dtype=np.float64).reshape(m, -1)
           for ch in _LOCKSTEP_CH}
    r_f = np.stack([np.asarray(bd[4]) for bd in blocks]).astype(np.float32)
    g_f = np.stack([np.asarray(bd[5]) for bd in blocks]).astype(np.float32)
    b_f = np.stack([np.asarray(bd[6]) for bd in blocks]).astype(np.float32)
    return _rgb_lockstep_core([bd[0] for bd in blocks], ZZ, ORG, r_f, g_f, b_f,
                              fallback_multipliers, base_accuracy, sf, dct_tp, config)


def _integer_refine_block(q_start, r8, g8, b8, sf, base_accuracy, max_iter=6):
    """Integer refinement of one cm=6 block for which no multiplier met the accuracy.

    q_start: dict channel -> int coefficient vector (zigzag, already round(c*sf)). The search moves
    single coefficients by +-1 (integers, so the stream format and the decoder are unchanged) and
    keeps a move when the maximum RGB error of the decoder-equivalent reconstruction drops.
    Returns (q, err) with the best vectors found."""
    n_edge = int(round(q_start["Y"].size ** 0.5))

    def err_of(qs):
        planes = [np.clip(idct2(from_zigzag(qs[ch] / sf, n_edge), 2) + 128.0, 0.0, 255.0)
                  for ch in ("Y", "Cb", "Cr")]
        R, G, B = ycbcr_to_rgb(*planes)
        return float(max(np.max(np.abs(r8 - R)), np.max(np.abs(g8 - G)), np.max(np.abs(b8 - B))))

    qs = {ch: np.asarray(q_start[ch], dtype=np.float64).copy() for ch in _LOCKSTEP_CH}
    best = err_of(qs)
    for _ in range(max_iter):
        improved = False
        for ch in _LOCKSTEP_CH:
            for k in range(qs[ch].size):
                for step in (-1.0, 1.0):
                    trial = {c: v.copy() for c, v in qs.items()}
                    trial[ch][k] += step
                    e = err_of(trial)
                    if e < best - 1e-12:
                        best, qs, improved = e, trial, True
        if not improved or best <= base_accuracy:
            break
    if best > base_accuracy:
        # single +-1 moves can stop in a local minimum: try pairs of moves
        pair = _pair_move(qs, r8, g8, b8, err_of, base_accuracy)
        if pair is not None:
            qs = pair
            best = err_of(qs)
            for _ in range(max_iter):
                improved = False
                for ch in _LOCKSTEP_CH:
                    for k in range(qs[ch].size):
                        for step in (-1.0, 1.0):
                            trial = {c: v.copy() for c, v in qs.items()}
                            trial[ch][k] += step
                            e = err_of(trial)
                            if e < best - 1e-12:
                                best, qs, improved = e, trial, True
                if not improved or best <= base_accuracy:
                    break
    return qs, best


def _pair_move(qs, r8, g8, b8, err_of, base_accuracy):
    """Vectorised search over all pairs of +-1 moves (3 channels x 64 coefficients). The reconstruction
    is linear in the coefficients, so each candidate is the current planes plus two rows of the IDCT
    basis. The first candidate that meets the accuracy is verified with err_of (the decoder's IDCT)
    before it is accepted. Returns the vectors or None."""
    nn = qs["Y"].size
    n = int(round(nn ** 0.5))
    if n > 8:  # 3 * n * n cells give millions of pairs; blocks left over fall back to a smaller N
        return None
    Bm = _get_idct_basis(n, 2).reshape(nn, nn)
    chs = list(_LOCKSTEP_CH)
    base_rec = np.stack([qs[ch] @ Bm for ch in chs])            # (3, nn)
    cells = np.array([(c, k) for c in range(3) for k in range(nn)])
    ii, jj = np.triu_indices(cells.shape[0], k=1)
    signs = np.array([(a, b) for a in (-1.0, 1.0) for b in (-1.0, 1.0)])
    for start in range(0, ii.size, 20000):
        sl = slice(start, min(start + 20000, ii.size))
        ci, ki = cells[ii[sl], 0], cells[ii[sl], 1]
        cj, kj = cells[jj[sl], 0], cells[jj[sl], 1]
        for sa, sb in signs:
            n = ci.size
            cand = np.broadcast_to(base_rec, (n, 3, nn)).copy()
            rows = np.arange(n)
            cand[rows, ci, :] += sa * Bm[ki]
            cand[rows, cj, :] += sb * Bm[kj]
            planes = np.clip(cand + 128.0, 0.0, 255.0)
            R, G, B = ycbcr_to_rgb(planes[:, 0].reshape(n, 8, 8), planes[:, 1].reshape(n, 8, 8),
                                   planes[:, 2].reshape(n, 8, 8))
            err = np.maximum(np.maximum(np.abs(r8 - R).max(axis=(1, 2)),
                                        np.abs(g8 - G).max(axis=(1, 2))),
                             np.abs(b8 - B).max(axis=(1, 2)))
            for idx in np.flatnonzero(err <= base_accuracy + 1e-9):
                trial = {ch: qs[ch].copy() for ch in chs}
                trial[chs[ci[idx]]][ki[idx]] += sa
                trial[chs[cj[idx]]][kj[idx]] += sb
                if err_of(trial) <= base_accuracy:
                    return trial
    return None


def _refine_singles_cpu(qs, err_of_batch, base_accuracy, max_iter=6):
    """Single-move loop of _integer_refine_block over a batch, bit-identical to it. Mutates qs."""
    K, nn = qs["Y"].shape
    best = err_of_batch(qs)
    alive = np.ones(K, dtype=bool)
    for _ in range(max_iter):
        improved = np.zeros(K, dtype=bool)
        for ch in _LOCKSTEP_CH:
            for k in range(nn):
                for step in (-1.0, 1.0):
                    saved = qs[ch][:, k].copy()
                    qs[ch][:, k] += step
                    e = err_of_batch(qs)
                    acc = alive & (e < best - 1e-12)
                    if acc.any():
                        best[acc] = e[acc]
                        improved |= acc
                        qs[ch][~acc, k] = saved[~acc]
                    else:
                        qs[ch][:, k] = saved
        alive &= improved & (best > base_accuracy)
        if not alive.any():
            break
    return best


def _integer_refine_batch(q_start, r8, g8, b8, sf, base_accuracy, max_iter=6):
    """Batched integer refinement over many blocks, bit-identical to _integer_refine_block.

    The single-move phase runs on all blocks at once (_refine_singles_cpu) with the per-block
    arithmetic: idct2 along axis 0 first (axis=1, then axis=2 on the (K, n, n) batch), bitwise
    equal to the per-block idct2. Blocks still above the accuracy need the pair-move phase and go
    through _integer_refine_block. Returns (q, err)."""
    K, nn = q_start["Y"].shape
    n = int(round(nn ** 0.5))
    zz = _get_zigzag_indices(n)
    from scipy.fft import idct as _idct

    def err_of_batch(qs):
        planes = []
        for ch in _LOCKSTEP_CH:
            flat = np.zeros((K, nn), dtype=np.float64)
            flat[:, zz] = qs[ch] / sf
            rec = _idct(_idct(flat.reshape(K, n, n), axis=1, norm="ortho", type=2),
                        axis=2, norm="ortho", type=2)
            planes.append(np.clip(rec + 128.0, 0.0, 255.0))
        R, G, B = ycbcr_to_rgb(*planes)
        return np.maximum(
            np.maximum(np.abs(r8 - R).max(axis=(1, 2)), np.abs(g8 - G).max(axis=(1, 2))),
            np.abs(b8 - B).max(axis=(1, 2)),
        )

    qs = {ch: np.asarray(q_start[ch], dtype=np.float64).copy() for ch in _LOCKSTEP_CH}
    best = _refine_singles_cpu(qs, err_of_batch, base_accuracy, max_iter)
    for k in np.flatnonzero(best > base_accuracy):
        q1, e1 = _integer_refine_block({ch: q_start[ch][k] for ch in _LOCKSTEP_CH},
                                       r8[k], g8[k], b8[k], sf, base_accuracy, max_iter)
        for ch in _LOCKSTEP_CH:
            qs[ch][k] = q1[ch]
        best[k] = e1
    return qs, best


def _rgb_lockstep_core(idx_list, ZZ, ORG, r_f, g_f, b_f, fallback_multipliers, base_accuracy, sf,
                       dct_tp, config, pred=None, scales=None):
    """Multiplier loop in lockstep on whole arrays. ZZ/ORG: dict channel -> (m, n*n) float64; r_f, g_f,
    b_f: (m, n, n) float32 originals; idx_list: block indices. pred (dict channel -> L_hat per block)
    with scales switches each channel search to the predicted-L acceptance (CPU). Returns
    [(idx, metadata, err)] or None."""
    from engine.u8_lockstep import search_rows, search_rows_pred

    m = len(idx_list)
    nn = ZZ[_LOCKSTEP_CH[0]].shape[1]
    n = int(round(nn ** 0.5))
    basis2 = np.ascontiguousarray(_get_idct_basis(n, dct_tp).reshape(nn, nn))
    basis3 = _get_idct_basis(n, dct_tp)
    bs = scaled_backscan(config.model.minimize_backscan, n)
    brk = int(config.model.backscan_break_after)
    verify = getattr(config.compression, "ycbcr_decode_verify", True)

    best_err = np.full(m, np.inf)
    best_mi = np.full(m, len(fallback_multipliers) - 1, dtype=np.int64)
    resolved = np.zeros(m, dtype=bool)
    bestL = {ch: np.zeros(m, dtype=np.int64) for ch in _LOCKSTEP_CH}
    bestRES = {ch: np.zeros((m, nn), dtype=np.int64) for ch in _LOCKSTEP_CH}
    bestM = {ch: np.zeros(m) for ch in _LOCKSTEP_CH}

    # the whole batch (3 channels and the RGB originals) is copied to VRAM once; per multiplier only the
    # row indices and parameters are sent, and the RGB check reads the coefficients already in VRAM
    sess = None
    if pred is None and m >= GPU_MIN_ROWS and gpu_session.available(n):
        try:
            sess = gpu_session.GpuSession([ZZ[ch] for ch in _LOCKSTEP_CH],
                                          [ORG[ch] for ch in _LOCKSTEP_CH], basis2,
                                          rgb=(r_f, g_f, b_f))
        except RuntimeError as exc:
            gpu_u8.warn_once(f"GPU session failed, using CPU: {exc}")

    for mi, mult in enumerate(fallback_multipliers):
        U = np.flatnonzero(~resolved)
        if U.size == 0:
            break
        u = U.size
        acc = float(mult) * base_accuracy
        LF = {}
        RES = {}
        planes = {}
        on_gpu = sess is not None
        for ci, ch in enumerate(_LOCKSTEP_CH):
            if pred is not None:
                lf, er, rs = search_rows_pred(ZZ[ch], ORG[ch], U, np.full(u, acc),
                                              np.full(u, float(sf)),
                                              np.full(u, bs, dtype=np.int64), basis2,
                                              pred[ch][U], scales,
                                              brk=int(config.model.backscan_break_after))
                on_gpu = False
            else:
                lf, er, rs, used = search_rows(sess if on_gpu else None, ci, U, ZZ[ch], ORG[ch],
                                               np.full(u, acc), np.full(u, float(sf)),
                                               np.full(u, bs, dtype=np.int64), basis2, brk)
                on_gpu = on_gpu and used
            LF[ch], RES[ch] = lf, rs

        # pre-verification RGB error: GPU (rgb_pre.cu) or the numpy path. Both return the rows whose YCbCr
        # rounding is not robust (tie); they are recomputed exactly below
        err = tie = None
        if on_gpu:
            try:
                err, tie = sess.rgb(U, float(sf))
            except RuntimeError as exc:
                gpu_u8.warn_once(f"GPU session rgb failed, using CPU: {exc}")
        if err is None and u >= GPU_MIN_ROWS and gpu_rgb.available(n):
            try:
                err, tie = gpu_rgb.pre_err(RES["Y"], RES["Cb"], RES["Cr"], r_f[U], g_f[U], b_f[U],
                                           basis2, float(sf))
            except RuntimeError as exc:  # fallback on CPU, same result
                gpu_u8.warn_once(f"GPU rgb phase failed, using CPU: {exc}")
        if err is None:
            for ch in _LOCKSTEP_CH:
                cdq = RES[ch].astype(np.float64) / sf
                planes[ch] = np.clip((cdq @ basis2).reshape(u, n, n) + 128.0, 0.0, 255.0)
            err, tie = _pre_err_cpu(planes["Y"], planes["Cb"], planes["Cr"],
                                    r_f[U], g_f[U], b_f[U])
        # The pre-verification error gates the decoder check (err <= base). The matmul above can differ
        # from the per-block tensordot by a few ulps, which flips the gate for blocks exactly on the
        # threshold, so those blocks are recomputed with the loop arithmetic.
        if verify:
            near = np.flatnonzero(
                (np.abs(err - base_accuracy) <= 1e-9 * max(1.0, base_accuracy)) | tie
            )
            if near.size:
                # grouped by L: a tensordot over a stack of rows is bitwise equal to the per-row call
                ex = {ch: np.zeros((near.size, n, n)) for ch in _LOCKSTEP_CH}
                for ch in _LOCKSTEP_CH:
                    Lc = LF[ch][near]
                    for L_val in np.unique(Lc):
                        sel = np.flatnonzero(Lc == L_val)
                        if L_val > 0:
                            cdq_g = RES[ch][near[sel], :L_val].astype(np.float64) / sf
                            rec_g = np.tensordot(cdq_g, basis3[:L_val], axes=(1, 0))
                        else:
                            rec_g = np.zeros((sel.size, n, n), dtype=np.float64)
                        ex[ch][sel] = np.clip(rec_g + 128.0, 0.0, 255.0)
                R_e, G_e, B_e = ycbcr_to_rgb(ex["Y"], ex["Cb"], ex["Cr"])
                err[near] = np.maximum(
                    np.maximum(
                        np.max(np.abs(r_f[U[near]] - R_e), axis=(1, 2)),
                        np.max(np.abs(g_f[U[near]] - G_e), axis=(1, 2)),
                    ),
                    np.max(np.abs(b_f[U[near]] - B_e), axis=(1, 2)),
                )
        final = err.copy()
        if verify:
            sel = np.flatnonzero(err <= base_accuracy)
            if sel.size:
                final[sel] = _rgb_decoder_err_batch(
                    {ch: RES[ch][sel] for ch in _LOCKSTEP_CH},
                    {ch: LF[ch][sel] for ch in _LOCKSTEP_CH},
                    sf, r_f[U[sel]], g_f[U[sel]], b_f[U[sel]],
                )

        upd = final < best_err[U]
        iU = U[upd]
        best_err[iU] = final[upd]
        best_mi[iU] = mi
        for ch in _LOCKSTEP_CH:
            bestL[ch][iU] = LF[ch][upd]
            bestRES[ch][iU] = RES[ch][upd]
            bestM[ch][iU] = float(mult)
        resolved[U[final <= base_accuracy]] = True

    if sess is not None:
        sess.close()

    # Blocks that no multiplier brought within the accuracy get an integer refinement of the coefficients
    # (format and decoder unchanged). Only these blocks change.
    unresolved = np.flatnonzero(~resolved)
    if unresolved.size:
        q_starts = {ch: np.rint(ZZ[ch][unresolved] * sf) for ch in _LOCKSTEP_CH}
        q_ref, e_ref = _integer_refine_batch(
            q_starts, r_f[unresolved].astype(np.float64), g_f[unresolved].astype(np.float64),
            b_f[unresolved].astype(np.float64), sf, base_accuracy)
        for j, k in enumerate(unresolved):
            if e_ref[j] <= base_accuracy:
                q = {ch: q_ref[ch][j] for ch in _LOCKSTEP_CH}
                # A channel left without coefficients is stored with L = 0 (neutral 128 in the decoder).
                # If all three channels are empty, the block would look like the all-zero marker, which
                # decodes as black, so Y keeps one zero DC coefficient.
                all_empty = not any(np.any(q[ch]) for ch in _LOCKSTEP_CH)
                for ch in _LOCKSTEP_CH:
                    nz = np.flatnonzero(q[ch])
                    L_new = int(nz[-1]) + 1 if nz.size else 0
                    if all_empty and ch == _LOCKSTEP_CH[0]:
                        L_new = 1
                    bestRES[ch][k, :] = 0
                    bestRES[ch][k, :L_new] = q[ch][:L_new].astype(np.int64)
                    bestL[ch][k] = L_new
                best_err[k] = e_ref[j]
                resolved[k] = True

    if not np.all(np.isfinite(best_err)):
        return None

    results = []
    for k, bidx in enumerate(idx_list):
        meta = {}
        for ch in _LOCKSTEP_CH:
            L = int(bestL[ch][k])
            meta[ch] = {"L": L, "coeffs": bestRES[ch][k, :L].copy(), "multiplier": float(bestM[ch][k])}
        metadata = {
            "Y": meta["Y"],
            "Cb": meta["Cb"],
            "Cr": meta["Cr"],
            "sf": sf,
            "fallback": bool(best_mi[k] > 0),
            "masks": None,
        }
        results.append((bidx, metadata, float(best_err[k])))
    return results


def process_rgb_arrays(idx, Y_blk, Cb_blk, Cr_blk, r_blk, g_blk, b_blk, config):
    """cm=6 for whole arrays of 8x8 blocks (no per-block tuples): the same DCT, rounding and zigzag as
    process_block_batch_rgb, then the lockstep core. Returns [(idx, metadata, err)], or None when the
    array path does not apply (heuristic backend, predictor without accept, other block size) or a
    block stays unresolved; the caller then uses the tuple path. With uint8_use_L_prediction and
    accept_prediction_if_within_accuracy the per-channel L_hat drives the channel search."""
    if not _array_path_enabled(config):
        return None
    if config.model.backend.lower() in ("heuristic",) or config.model.advanced_heuristic:
        return None
    predict = bool(config.model.uint8_use_L_prediction)
    if predict and not config.model.accept_prediction_if_within_accuracy:
        return None
    N = len(idx)
    if N == 0:
        return []
    n = Y_blk.shape[1]
    if Y_blk.shape[1:] != (n, n) or n not in ARRAY_EDGES or (predict and n != 8):
        return None
    from scipy.fft import dct as scipy_dct

    fallback_multipliers = list(getattr(config.compression, "ycbcr_fallback_multipliers",
                                        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2]))
    base_accuracy = float(getattr(config.compression, "uint8_accuracy", 5))
    dec = config.compression.decimal_places
    zz_idx = _get_zigzag_indices(n)
    ZZ, ORG = {}, {}
    for name, blk in (("Y", Y_blk), ("Cb", Cb_blk), ("Cr", Cr_blk)):
        A = blk.astype(np.float64) - 128.0
        d = np.round(scipy_dct(scipy_dct(A, axis=1, norm="ortho"), axis=2, norm="ortho"), dec)
        ZZ[name] = np.ascontiguousarray(d.reshape(N, -1)[:, zz_idx])
        ORG[name] = np.ascontiguousarray(A.reshape(N, n * n))
    pred = scales = None
    if predict:
        Lh = predict_L_arrays(np.concatenate([ZZ["Y"], ZZ["Cb"], ZZ["Cr"]]), config,
                              base_accuracy, is_ycbcr=True).reshape(3, N)
        pred = {"Y": Lh[0], "Cb": Lh[1], "Cr": Lh[2]}
        scales = tuple(float(x) for x in config.model.uint8_L_prediction_scales)
    return _rgb_lockstep_core(
        list(idx), ZZ, ORG, r_blk.astype(np.float32), g_blk.astype(np.float32),
        b_blk.astype(np.float32), fallback_multipliers, base_accuracy, config.scaling_factor,
        config.compression.dct_type, config, pred=pred, scales=scales)


def process_block_batch_rgb(
    batch_data: Tuple[int, List],
    config: Optional[Config] = None,
) -> List[Tuple[int, Any, float]]:
    """Process RGB block triplets with per-block YCbCr multiplier selection (cm=6).

    Each input block: (idx, Y, Cb, Cr, r, g, b, nodata)
    Returns: [(idx, metadata, max_rgb_error), ...]

    For each block, we predict the starting multiplier from chroma amplitude and
    iterate DOWN only when needed.  Typical case: 1 attempt per block.

    metadata format (compatible with serialize_compressed_blocks_rgb):
        {
          'Y':  {'L': int, 'coeffs': np.ndarray, 'multiplier': float},
          'Cb': {'L': int, 'coeffs': np.ndarray, 'multiplier': float},
          'Cr': {'L': int, 'coeffs': np.ndarray, 'multiplier': float},
          'sf': int,
          'fallback': bool,   # True if start_idx prediction was not sufficient
          'masks': None,
        }
    """
    if config is None:
        config = load_config()

    _batch_idx, blocks = batch_data

    fallback_multipliers: list = list(
        getattr(
            config.compression,
            "ycbcr_fallback_multipliers",
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2],
        )
    )
    base_accuracy = float(getattr(config.compression, "uint8_accuracy", 5))
    sf = config.scaling_factor
    dct_tp = config.compression.dct_type
    dec = config.compression.decimal_places
    n_mult = len(fallback_multipliers)

    use_predictor = (
        config.model.backend.lower() in ("heuristic",)
        or config.model.advanced_heuristic
        or config.model.uint8_use_L_prediction
    )

    L_predictions = {}  # idx -> {'Y': L_y, 'Cb': L_cb, 'Cr': L_cr}
    predictor = None

    if use_predictor:
        try:
            # one predictor for all block sizes
            n = blocks[0][1].shape[1] if blocks else 16

            from predictor.predictor import get_predictor

            cache_key = (
                config.model.backend.lower(),
                config.model.advanced_heuristic,
                base_accuracy,
                n,
                "uint8",  # cm=6 is always uint8
                True,  # is_ycbcr=True
            )
            if cache_key not in _predictor_cache:
                _predictor_cache[cache_key] = get_predictor(
                    backend=config.model.backend.lower(),
                    advanced_heuristic=config.model.advanced_heuristic,
                    accuracy=base_accuracy,
                    block_size=n,
                    config=config,
                    src_dtype="uint8",
                    is_ycbcr=True,
                )
            predictor = _predictor_cache[cache_key]

        except (ImportError, AttributeError, RuntimeError) as e:
            if not config.output.quiet:
                print(f"  Warning: Predictor initialization failed for cm=6: {e}")

    results: List[Tuple[int, Any, float]] = []

    # Pre-computation for all blocks before the refinement loop: batch DCT, zigzag and L prediction.
    # Blocks may have different sizes (auto_select_block_size), so they are grouped by n.
    n_blocks = len(blocks)
    if n_blocks == 0:
        return results

    # group the blocks by block size (8, 16 or 32)
    from collections import defaultdict

    size_groups = defaultdict(list)  # n -> [(k, block_data), ...]
    for k, bd in enumerate(blocks):
        bn = bd[1].shape[0]
        size_groups[bn].append((k, bd))

    from scipy.fft import dct as scipy_dct

    # result arrays, indexed by the original position k
    all_zigzag_Y = [None] * n_blocks
    all_zigzag_Cb = [None] * n_blocks
    all_zigzag_Cr = [None] * n_blocks
    all_centered_Y = [None] * n_blocks
    all_centered_Cb = [None] * n_blocks
    all_centered_Cr = [None] * n_blocks
    L_predictions = {}

    # batch DCT and zigzag
    def _batch_dct_zigzag(blocks_arr, zigzag_idx, gn):
        dct_batch = scipy_dct(scipy_dct(blocks_arr, axis=1, norm="ortho"), axis=2, norm="ortho")
        dct_batch = np.round(dct_batch, dec)
        return dct_batch.reshape(gn, -1)[:, zigzag_idx]

    for bn, group in size_groups.items():
        gn = len(group)
        zigzag_idx = _get_zigzag_indices(bn)

        # stack the channel blocks of this size group
        Y_all = np.stack([bd[1] for _, bd in group]).astype(np.float64) - 128.0
        Cb_all = np.stack([bd[2] for _, bd in group]).astype(np.float64) - 128.0
        Cr_all = np.stack([bd[3] for _, bd in group]).astype(np.float64) - 128.0

        # batch DCT and zigzag
        zz_Y = _batch_dct_zigzag(Y_all, zigzag_idx, gn)
        zz_Cb = _batch_dct_zigzag(Cb_all, zigzag_idx, gn)
        zz_Cr = _batch_dct_zigzag(Cr_all, zigzag_idx, gn)

        # batch L prediction
        if use_predictor and predictor is not None:
            try:
                # predictor for this block size
                cache_key_bn = (
                    config.model.backend.lower(),
                    config.model.advanced_heuristic,
                    base_accuracy,
                    bn,
                    "uint8",
                )
                if cache_key_bn not in _predictor_cache:
                    from predictor.predictor import get_predictor

                    _predictor_cache[cache_key_bn] = get_predictor(
                        backend=config.model.backend.lower(),
                        advanced_heuristic=config.model.advanced_heuristic,
                        accuracy=base_accuracy,
                        block_size=bn,
                        config=config,
                        src_dtype="uint8",
                        is_ycbcr=True,
                    )
                predictor_bn = _predictor_cache[cache_key_bn]

                all_zz = np.concatenate([zz_Y, zz_Cb, zz_Cr], axis=0)
                all_preds, _ = predictor_bn.predict_from_dct_batch(all_zz)

                for gi, (k, _) in enumerate(group):
                    L_predictions[blocks[k][0]] = {
                        "Y": int(all_preds[gi]),
                        "Cb": int(all_preds[gn + gi]),
                        "Cr": int(all_preds[2 * gn + gi]),
                    }
            except (ValueError, IndexError, RuntimeError):
                pass

        # store into the per-block arrays
        for gi, (k, _) in enumerate(group):
            all_zigzag_Y[k] = zz_Y[gi]
            all_zigzag_Cb[k] = zz_Cb[gi]
            all_zigzag_Cr[k] = zz_Cr[gi]
            all_centered_Y[k] = Y_all[gi]
            all_centered_Cb[k] = Cb_all[gi]
            all_centered_Cr[k] = Cr_all[gi]

    if (
        _array_path_enabled(config)
        and not use_predictor
        and _same_array_edge(blocks, lambda bd: bd[1])
    ):
        lock = _rgb_batch_lockstep(
            blocks, {"Y": all_zigzag_Y, "Cb": all_zigzag_Cb, "Cr": all_zigzag_Cr},
            {"Y": all_centered_Y, "Cb": all_centered_Cb, "Cr": all_centered_Cr},
            fallback_multipliers, base_accuracy, sf, dct_tp, config,
        )
        if lock is not None:
            return lock

    # Per-block refinement. For each block the multipliers are tried from the largest. When the per-channel
    # errors satisfy the analytical bound (bound <= accuracy), the IDCT and YCbCr-to-RGB reconstruction is
    # skipped and the block is accepted.

    for k, block_data in enumerate(blocks):
        idx, _Y_b, _Cb_b, _Cr_b, r_b, g_b, b_b, _nodata = block_data
        n = _Y_b.shape[0]

        zigzag_blocks = {
            "Y": all_zigzag_Y[k],
            "Cb": all_zigzag_Cb[k],
            "Cr": all_zigzag_Cr[k],
        }
        centered_blocks = {
            "Y": all_centered_Y[k],
            "Cb": all_centered_Cb[k],
            "Cr": all_centered_Cr[k],
        }

        start_idx = 0  # always try the largest multiplier first: smallest archive that meets the accuracy

        # original RGB converted to float32 once
        r_f = r_b.astype(np.float32)
        g_f = g_b.astype(np.float32)
        b_f = b_b.astype(np.float32)

        best_metadata: Optional[Dict] = None
        best_err = float("inf")
        best_mult_idx = n_mult - 1
        # err(L) per channel does not depend on the multiplier (only the threshold changes)
        err_caches = {"Y": {}, "Cb": {}, "Cr": {}}

        for mult_idx in range(start_idx, n_mult):
            mult = fallback_multipliers[mult_idx]
            ch_acc = mult * base_accuracy

            try:
                ch_data: Dict[str, Dict] = {}

                for ch_name in ("Y", "Cb", "Cr"):
                    zigzag = zigzag_blocks[ch_name]
                    centered = centered_blocks[ch_name]
                    L_pred = (
                        L_predictions.get(idx, {}).get(ch_name) if idx in L_predictions else None
                    )

                    compressed_arr, _, _ = refine_dct_array(
                        zigzag,
                        ch_acc,
                        centered,
                        n,
                        dct_tp,
                        sf,
                        config,
                        predicted_L=L_pred,
                        src_dtype="uint8",
                        err_cache=err_caches[ch_name],
                    )

                    L = len(compressed_arr)
                    ch_data[ch_name] = {
                        "L": L,
                        "coeffs": compressed_arr,
                        "multiplier": mult,
                    }

                # full reconstruction to measure the actual error, with the tensordot basis (same arithmetic as
                # reconstruct_block).
                basis = _get_idct_basis(n, dct_tp)
                ch_recons: Dict[str, np.ndarray] = {}
                for ch_name in ("Y", "Cb", "Cr"):
                    L = ch_data[ch_name]["L"]
                    compressed_arr = ch_data[ch_name]["coeffs"]
                    if L > 0:
                        coeffs_dq = compressed_arr.astype(np.float64) / sf
                        recon = np.tensordot(coeffs_dq, basis[:L], axes=(0, 0))
                    else:
                        recon = np.zeros((n, n), dtype=np.float64)
                    ch_recons[ch_name] = np.clip(recon + 128.0, 0.0, 255.0)

                Y_r = ch_recons["Y"]
                Cb = ch_recons["Cb"]
                Cr = ch_recons["Cr"]

                R_rec, G_rec, B_rec = ycbcr_to_rgb(Y_r, Cb, Cr)

                max_rgb_err = float(
                    max(
                        np.max(np.abs(r_f - R_rec)),
                        np.max(np.abs(g_f - G_rec)),
                        np.max(np.abs(b_f - B_rec)),
                    )
                )

                if max_rgb_err <= base_accuracy and getattr(
                    config.compression, "ycbcr_decode_verify", True
                ):
                    max_rgb_err = _rgb_error_via_decoder(ch_data, n, sf, r_f, g_f, b_f)

                if max_rgb_err < best_err:
                    best_err = max_rgb_err
                    best_metadata = ch_data
                    best_mult_idx = mult_idx

                if max_rgb_err <= base_accuracy:
                    break  # accepted — stop iterating

            except (ValueError, IndexError, RuntimeError):
                continue

        if best_metadata is None:
            # Every multiplier raised an exception. The error must not be recorded as 0, which would hide the
            # violation. A DC-only block decodes to a flat plane; its error is measured through the decoder path.
            mult = fallback_multipliers[-1]
            best_metadata = {}
            for ch_name in ("Y", "Cb", "Cr"):
                zz = np.asarray(zigzag_blocks[ch_name], dtype=np.float64)
                coeffs = (
                    np.round(zz[:1] * sf).astype(np.int64)
                    if zz.size
                    else np.array([], dtype=np.int64)
                )
                best_metadata[ch_name] = {
                    "L": int(coeffs.size),
                    "coeffs": coeffs,
                    "multiplier": mult,
                }
            best_err = _rgb_error_via_decoder(best_metadata, n, sf, r_f, g_f, b_f)
            best_mult_idx = n_mult - 1

        metadata = {
            "Y": best_metadata["Y"],
            "Cb": best_metadata["Cb"],
            "Cr": best_metadata["Cr"],
            "sf": sf,
            "fallback": best_mult_idx > start_idx,
            "masks": None,
        }
        results.append((idx, metadata, best_err))

    return results
