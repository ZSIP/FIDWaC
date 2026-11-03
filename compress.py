#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIDWAC v2 - CLI
================
"""

import os
import re
import sys
import argparse
import time
import subprocess

# One BLAS thread per worker process. These variables must be set before numpy is imported,
# which happens when compress.compression is loaded.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config, Config, set_gpu_mode
from compress.compression import compress_image
from compress.decompression import decompress_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FIDWAC v2 - DCT compression for geospatial data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compress.py -i input.tif -o output.7z               # Compression
  python compress.py -i input.png -o output.7z               # PNG + world file
  python compress.py -i input.tif -o output.7z --auto        # Auto-select block size
  python compress.py -i input.tif -o output.7z -a 0.01       # High accuracy
  python compress.py -i compressed.7z -o output.tif          # Decompression
  python compress.py -i input.tif -o output.7z --backend heuristic  # Use heuristic prediction

Supported input formats: GeoTIFF, PNG+PNGW, JPEG+JGW, BMP+BPW, ASC, Erdas Imagine (.img)
        """,
    )

    parser.add_argument("-i", "--input", required=True, help="Input file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file (optional for decompression: auto-derived from input)",
    )
    parser.add_argument("--config", "-c", help="Config file path")
    parser.add_argument("--accuracy", "-a", type=float, help="Compression accuracy (e.g., 0.05)")
    parser.add_argument(
        "--method",
        choices=["dct", "pred", "sz3"],
        default="dct",
        help="dct = block DCT (cm=3/5/6, default); pred = near-lossless MED prediction (cm=7, .fpred); sz3 = external SZ3 absolute-error path (cm=8, .sz3)",
    )
    parser.add_argument("--block-size", "-n", type=int, choices=[8, 16, 32], help="Block size")
    parser.add_argument("--auto", action="store_true", help="Auto-select block size")
    parser.add_argument("--backend", choices=["binary", "heuristic"], help="Prediction backend")
    parser.add_argument(
        "--advanced-heuristic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable advanced heuristic (lookup table, 16 features)",
    )
    parser.add_argument(
        "--minimize-backscan", type=int, default=None, help="Minimize backscan threshold"
    )
    parser.add_argument(
        "--backscan-break-after",
        type=int,
        default=None,
        help="Break backscan after N consecutive failures (0 = never)",
    )
    parser.add_argument(
        "--accept-prediction",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Accept prediction when within accuracy tolerance "
        "(skip binary search for float/uint8)",
    )
    parser.add_argument(
        "--decimal-places",
        type=int,
        default=None,
        help="Number of decimal places for scaling (default: 2)",
    )
    parser.add_argument("--processes", "-p", type=int, help="Number of processes")
    parser.add_argument(
        "--gpu",
        choices=["auto", "on", "off"],
        default=None,
        help="GPU batch path: auto (GPU when available, else CPU with a note), on (required), off (CPU)",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    parser.add_argument(
        "--uint8-accuracy-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="uint8 accuracy mode (binary search)",
    )
    parser.add_argument(
        "--uint8-accuracy",
        type=int,
        default=None,
        help="Max color value error for uint8 accuracy (min=2, due to YCbCr rounding)",
    )
    parser.add_argument(
        "--rgb-quality", type=int, default=None, help="JPEG quality for RGB (1-100)"
    )
    parser.add_argument(
        "--lossless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Lossless PNG mode for uint8",
    )
    parser.add_argument(
        "--rgb-channel-indices",
        type=str,
        default=None,
        help="RGB channel indices R,G,B (1-based, e.g. 1,2,3)",
    )
    parser.add_argument(
        "--scaling-factor",
        type=str,
        default=None,
        help="Scaling factor or list of factors (comma-separated, e.g. 1,10)",
    )
    parser.add_argument(
        "--ycbcr-y-multiplier",
        type=float,
        default=None,
        help="YCbCr Y channel accuracy multiplier (default 1.0)",
    )
    parser.add_argument(
        "--ycbcr-cb-multiplier",
        type=float,
        default=None,
        help="YCbCr Cb channel accuracy multiplier (default 1.0)",
    )
    parser.add_argument(
        "--ycbcr-cr-multiplier",
        type=float,
        default=None,
        help="YCbCr Cr channel accuracy multiplier (default 1.0)",
    )
    parser.add_argument(
        "--ycbcr-per-block",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable per-block YCbCr multiplier selection " "(validity checked in RGB space)",
    )
    parser.add_argument(
        "--allowed-block-sizes",
        type=str,
        default=None,
        help="Allowed block sizes e.g. 8,16,32",
    )
    parser.add_argument(
        "--crs",
        type=str,
        default=None,
        help="CRS of the output: empty = CRS of the input file (default), 'none' = no CRS, or e.g. epsg:2180",
    )
    parser.add_argument(
        "--compression-method",
        type=str,
        default=None,
        choices=["LZMA2", "PPMD", "BZIP2", "DEFLATE"],
        help="7z compression method",
    )
    parser.add_argument(
        "--lzma2-block-size",
        type=str,
        default=None,
        help="LZMA2 block size of the .7z archive, e.g. 1m, 2m, 4m ('' = 7-Zip default); "
        "smaller blocks compress faster with nearly the same size",
    )
    parser.add_argument(
        "--verify-with-full-idct",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Validate final block error using full IDCT reconstruction",
    )
    parser.add_argument(
        "--l2-precheck",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable L2 precheck for fast rejection",
    )
    parser.add_argument(
        "--incremental-backscan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable incremental backscan",
    )
    parser.add_argument(
        "--uint8-L-prediction-scale",
        type=float,
        default=None,
        help="Single scale factor for uint8 L prediction (legacy shortcut)",
    )
    parser.add_argument(
        "--uint8-L-prediction-scales",
        type=str,
        default=None,
        help="Comma-separated uint8 L prediction cascade, " "e.g. 1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0",
    )

    return parser


def _apply_args_to_config(args: argparse.Namespace, config: Config) -> None:
    """Apply CLI argument overrides to the config object."""
    c, m, p, o = config.compression, config.model, config.performance, config.output
    if getattr(args, "gpu", None) is not None:
        p.gpu = args.gpu
        set_gpu_mode(args.gpu)

    # Compression settings
    # --method always sets the method (default: dct). The method saved in config.json by the web GUI
    # does not change a command-line run.
    c.method = args.method
    if args.accuracy is not None:
        c.accuracy = args.accuracy
    if args.block_size is not None:
        c.block_size = args.block_size
        c.auto_select_block_size = False
    if args.auto:
        c.auto_select_block_size = True
    if args.decimal_places is not None:
        c.decimal_places = args.decimal_places
    if args.uint8_accuracy_mode is not None:
        c.uint8_accuracy_mode = args.uint8_accuracy_mode
    if args.uint8_accuracy is not None:
        c.uint8_accuracy = args.uint8_accuracy
    if args.rgb_quality is not None:
        c.rgb_quality = args.rgb_quality
    if args.lossless is not None:
        c.lossless = args.lossless
    if args.rgb_channel_indices is not None:
        c.rgb_channel_indices = [int(x) for x in args.rgb_channel_indices.split(",")]
    if args.scaling_factor is not None:
        sf_str = args.scaling_factor.strip("[]").strip()
        if "," in sf_str:
            c.uint8_scaling_factor = [int(x.strip()) for x in sf_str.split(",") if x.strip()]
        else:
            c.uint8_scaling_factor = int(sf_str)
    if args.ycbcr_y_multiplier is not None:
        c.ycbcr_y_multiplier = args.ycbcr_y_multiplier
    if args.ycbcr_cb_multiplier is not None:
        c.ycbcr_cb_multiplier = args.ycbcr_cb_multiplier
    if args.ycbcr_cr_multiplier is not None:
        c.ycbcr_cr_multiplier = args.ycbcr_cr_multiplier
    if args.ycbcr_per_block is not None:
        c.ycbcr_per_block_mode = args.ycbcr_per_block
    if args.allowed_block_sizes is not None:
        c.allowed_block_sizes = [int(x) for x in args.allowed_block_sizes.split(",")]
    if args.crs is not None:
        c.crs = args.crs

    if args.backend is not None:
        m.backend = args.backend
    if args.advanced_heuristic is not None:
        m.advanced_heuristic = args.advanced_heuristic
    if args.minimize_backscan is not None:
        m.minimize_backscan = args.minimize_backscan
    if args.backscan_break_after is not None:
        m.backscan_break_after = args.backscan_break_after
    if args.accept_prediction is not None:
        m.accept_prediction_if_within_accuracy = args.accept_prediction
    if args.uint8_L_prediction_scales is not None:
        m.uint8_L_prediction_scales = [
            float(x.strip()) for x in args.uint8_L_prediction_scales.split(",") if x.strip()
        ]
    if args.uint8_L_prediction_scale is not None:
        m.uint8_L_prediction_scales = [args.uint8_L_prediction_scale]

    if args.compression_method is not None:
        o.compression_method = args.compression_method
    if args.lzma2_block_size is not None:
        o.lzma2_block_size = args.lzma2_block_size
    if args.quiet:
        o.quiet = True

    if args.verify_with_full_idct is not None:
        p.verify_with_full_idct = args.verify_with_full_idct
    if args.l2_precheck is not None:
        p.l2_precheck_enabled = args.l2_precheck
    if args.incremental_backscan is not None:
        p.incremental_backscan = args.incremental_backscan


def _gpu_free_vram_mb():
    """Free VRAM of the first GPU in MB (nvidia-smi, no CUDA context), or None when unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return int(out.stdout.strip().splitlines()[0].split(",")[0].strip())
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def _probe_image_dims(path):
    """(height, width) from a PNG/TIFF header without decoding the pixels, or None."""
    if not path:
        return None
    try:
        with open(path, "rb") as fh:
            head = fh.read(32)
        if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
            return int.from_bytes(head[20:24], "big"), int.from_bytes(head[16:20], "big")
    except OSError:
        pass
    try:
        import rasterio
        with rasterio.open(path) as src:
            return src.height, src.width
    except Exception:
        return None


# One CUDA context (runtime + kernels) per worker process of the block Pool, and the per-strip
# sessions together hold about one file-sized batch: 3x ZZ + 3x ORG float64 + 3x RGB float32.
_GPU_CTX_MB = 500
_GPU_BYTES_PER_ROW = 64 * (6 * 8 + 3 * 4)


def _gpu_process_cap(input_path, max_cpus):
    """Auto-process cap when the GPU path is active: p contexts plus the file batch must fit in
    free VRAM (with headroom), otherwise sessions fail and blocks silently fall back to the CPU."""
    from engine.gpu_u8 import phases
    if not phases.available():
        return None
    free = _gpu_free_vram_mb()
    if free is None:
        return None
    session_mb = 0.0
    dims = _probe_image_dims(input_path)
    if dims is not None:
        h, w = dims
        rows = ((h + 7) // 8) * ((w + 7) // 8)  # N=8 yields the largest block count
        session_mb = rows * _GPU_BYTES_PER_ROW * 1.3 / 1e6
    budget = free * 0.75 - session_mb
    if budget < _GPU_CTX_MB:
        return 1
    return max(1, min(max_cpus, int(budget // _GPU_CTX_MB)))


def _resolve_num_processes(args, input_path=None):
    max_cpus = os.cpu_count() or 1
    if args.processes is not None:
        if args.processes < 1 or args.processes > max_cpus:
            print(f"Error: --processes must be between 1 and {max_cpus}")
            sys.exit(1)
        return args.processes
    if getattr(args, "gpu", None) != "off":
        cap = _gpu_process_cap(input_path, max_cpus)
        if cap is not None and cap < max_cpus:
            if not getattr(args, "quiet", False):
                print(f"Auto processes: {cap} (GPU VRAM cap; CPUs: {max_cpus})")
            return cap
    return max_cpus


def _extract_validity_tag(result_path: str) -> str:
    bn = os.path.basename(result_path)
    if "_VT." in bn:
        return "VT"
    if "_VF." in bn:
        return "VF"
    m = re.search(r"_Q(\d+)[_.]", bn)
    return f"Q{m.group(1)}" if m else "OK"


def _tag_output(path: str, tag: str) -> str:
    """Insert the validity tag before the extension: name.ext becomes name_<tag>.ext."""
    stem, ext = os.path.splitext(path)
    tagged = f"{stem}_{tag}{ext}"
    os.replace(path, tagged)
    return tagged


def _print_banner(operation: str, input_path: str, output_path: str, config: Config) -> None:
    print("=" * 60)
    print(f"FIDWAC v2 - {operation}")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    if operation == "Compression":
        print(f"Auto-select: {config.compression.auto_select_block_size}")
        print(f"Backend: {config.model.backend}")
        if config.compression.lossless:
            print("Accuracy: lossless (max error = 0)")
        elif config.compression.uint8_accuracy_mode:
            print(f"Accuracy: {config.compression.accuracy} (float) / {config.compression.uint8_accuracy} (uint8)")
        else:
            print(f"Accuracy: {config.compression.accuracy}")
    print("=" * 60)


def main():
    """Main entry point for FIDWAC v2 CLI."""
    args = _build_parser().parse_args()

    # Resolve output path: derive output dir when -o is a dir or absent.
    # output_dir_only = True  → compress_image/decompress_file decides the filename
    #                            (keeps full descriptive name with params)
    # output_dir_only = False → user gave an explicit filename → rename to it
    input_is_archive = args.input.lower().endswith((".7z", ".fpred", ".sz3"))
    output_dir_only = False

    if args.output is None:
        # no -o: use directory of input file, let compress/decompress name it
        args.output = os.path.dirname(os.path.abspath(args.input)) or "."
        output_dir_only = True
    elif os.path.isdir(args.output) or args.output.endswith(("/", os.sep)):
        # -o is an existing directory or trailing-slash path
        os.makedirs(args.output, exist_ok=True)
        output_dir_only = True
    # else: user gave explicit filename → output_dir_only stays False

    config = load_config(args.config)
    _apply_args_to_config(args, config)
    num_processes = _resolve_num_processes(args, args.input)

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    output_path = args.output
    output_dir = output_path if output_dir_only else (os.path.dirname(output_path) or ".")
    if not os.path.exists(output_dir):
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)

    start_time = time.time()

    # Raster inputs are compressed; archives (.7z, .fpred, .sz3) are decompressed.
    if output_dir_only:
        output_is_archive = not input_is_archive
    else:
        output_is_archive = output_path.lower().endswith((".7z", ".fpred", ".sz3"))

    if not output_dir_only:
        if input_is_archive and output_is_archive:
            print("Error: Both input and output are archives - cannot determine operation")
            sys.exit(1)
        if not input_is_archive and not output_is_archive:
            print(
                "Error: Neither input nor output is an archive (.7z) - "
                "cannot determine operation"
            )
            sys.exit(1)

    operation = "Compression" if output_is_archive else "Decompression"
    out_dir_for_func = output_path if output_dir_only else (os.path.dirname(output_path) or ".")

    if not config.output.quiet:
        _print_banner(operation, input_path, output_path, config)

    try:
        method = config.compression.method or "dct"
        if output_is_archive and method == "pred":
            from compress.compression_pred import compress_pred_archive

            eps_int = (
                config.compression.uint8_accuracy
                if config.compression.uint8_accuracy_mode
                else None
            )
            if output_path.lower().endswith(".7z"):
                result_path, max_err, valid = compress_pred_archive(
                    input_path,
                    output_path,
                    config.compression.accuracy,
                    quiet=config.output.quiet,
                    crs=config.compression.crs,
                    eps_int=eps_int,
                )
                tag = "VT" if valid else "VF"
                print(f"VALIDITY:{tag}", flush=True)
            else:
                base = os.path.splitext(os.path.basename(input_path))[0]
                result_path = os.path.join(
                    out_dir_for_func,
                    f"{base}_PRED_eps{config.compression.accuracy}.7z",
                )
                result_path, max_err, valid = compress_pred_archive(
                    input_path,
                    result_path,
                    config.compression.accuracy,
                    quiet=config.output.quiet,
                    crs=config.compression.crs,
                    eps_int=eps_int,
                )
                tag = "VT" if valid else "VF"
                result_path = _tag_output(result_path, tag)
                print(f"VALIDITY:{tag}", flush=True)
        elif output_is_archive and method == "sz3":
            from compress.compression_sz3 import compress_sz3_archive

            eps_int = (
                config.compression.uint8_accuracy
                if config.compression.uint8_accuracy_mode
                else None
            )
            if output_path.lower().endswith(".7z"):
                result_path, max_err, valid = compress_sz3_archive(
                    input_path,
                    output_path,
                    config.compression.accuracy,
                    quiet=config.output.quiet,
                    crs=config.compression.crs,
                    eps_int=eps_int,
                )
                tag = "VT" if valid else "VF"
                print(f"VALIDITY:{tag}", flush=True)
            else:
                base = os.path.splitext(os.path.basename(input_path))[0]
                result_path = os.path.join(
                    out_dir_for_func,
                    f"{base}_SZ3_eps{config.compression.accuracy}.7z",
                )
                result_path, max_err, valid = compress_sz3_archive(
                    input_path,
                    result_path,
                    config.compression.accuracy,
                    quiet=config.output.quiet,
                    crs=config.compression.crs,
                    eps_int=eps_int,
                )
                tag = "VT" if valid else "VF"
                result_path = _tag_output(result_path, tag)
                print(f"VALIDITY:{tag}", flush=True)
        elif output_is_archive:
            result_path = compress_image(input_path, config, num_processes, out_dir_for_func)
            print(f"VALIDITY:{_extract_validity_tag(result_path)}", flush=True)
        elif input_path.lower().endswith((".fpred", ".sz3")):
            if input_path.lower().endswith(".fpred"):
                from compress.compression_pred import decompress_pred

                base = os.path.splitext(os.path.basename(input_path))[0]
                result_path = decompress_pred(input_path, os.path.join(out_dir_for_func, f"{base}.tif"))
            else:
                from compress.compression_sz3 import decompress_sz3

                base = os.path.splitext(os.path.basename(input_path))[0]
                result_path = decompress_sz3(input_path, os.path.join(out_dir_for_func, f"{base}.tif"))
        else:
            result_path = decompress_file(input_path, config, out_dir_for_func)

        # Rename only when user gave an explicit output filename
        if not output_dir_only and result_path != output_path:
            os.replace(result_path, output_path)
        else:
            output_path = result_path
    except (IOError, OSError, RuntimeError, ValueError) as e:
        print(f"Error during {operation.lower()}: {e}")
        sys.exit(1)

    if not config.output.quiet:
        elapsed = time.time() - start_time
        print(f"\n{operation} completed in {elapsed:.2f}s")
    # Printed even with --quiet: the web GUI parses this line to locate the result.
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
