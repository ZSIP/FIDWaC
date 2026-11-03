#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIDWAC v2 - IDW interpolation of scattered points to GeoTIFF
============================================================
Algorithm ported from ZSIP/FIDWaC (MIT License, Copyright (c) 2025 ZSIP).
Parameters default to config.json, section "interpolation"; options override them.

Examples:
  python3 interpolate.py -i points.csv -o results/
  python3 interpolate.py -i points.las -o results/ --las-classes 2 --csv --png
  python3 interpolate.py -i tiles/ -o results/ --geoid --geoid-file geoid.txt
  python3 interpolate.py -i points.shp -o results/ --compress --method dct --accuracy 0.05
"""

import argparse
import dataclasses
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import load_config  # noqa: E402
from interpolation import EXTRA_OUTPUTS, POINT_EXTENSIONS, run_interpolation  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="IDW interpolation of scattered points to a GeoTIFF (ZSIP/FIDWaC algorithm).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True,
                        help="point file (TXT/CSV, LAS/LAZ, SHP) or a folder with such files")
    parser.add_argument("-o", "--output", required=True, help="output folder")
    parser.add_argument("--config", "-c", help="config.json path (default: config.json in the project folder)")
    parser.add_argument("-N", "--neighbours", type=int, help="nearest points per cell (default 10)")
    parser.add_argument("--resolution", type=float, help="cell size, in units of --source-crs (default 0.5)")
    parser.add_argument("--max-distance", type=float, help="search radius, in units of --source-crs (default 5)")
    parser.add_argument("--weights", type=float, help="power p of the IDW weights 1 / d^p (default 2)")
    parser.add_argument("--leafsize", type=int, help="KD-tree leaf size (default 10)")
    parser.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=None,
                        help="Gaussian smoothing of the grid (default: on)")
    parser.add_argument("--smooth-level", type=float, help="Gaussian sigma in cells (default 1)")
    parser.add_argument("--r-min-function", action=argparse.BooleanOptionalAction, default=None,
                        help="reduce distances below --r-min (default: off)")
    parser.add_argument("--r-min", type=float, help="threshold for --r-min-function (default 0.5)")
    parser.add_argument("--las-classes", help="keep only these LAS/LAZ classes, for example 2,6")
    parser.add_argument("--z-field", help="z attribute of SHP files (default z)")
    parser.add_argument("--geoid", action=argparse.BooleanOptionalAction, default=None,
                        help="subtract the geoid undulation from z (default: off)")
    parser.add_argument("--geoid-file", help="geoid file, lines of x y z")
    parser.add_argument("--geoid-crs", help="CRS of the geoid file (default epsg:4326)")
    parser.add_argument("--source-crs", help="CRS of the points and of the output (default epsg:2180)")
    parser.add_argument("--geoid-max-distance", type=float,
                        help="geoid points farther than this are not used (default 2000)")
    parser.add_argument("--csv", action=argparse.BooleanOptionalAction, default=None,
                        help="also write x,y,value CSV (values before smoothing)")
    parser.add_argument("--surfer", action=argparse.BooleanOptionalAction, default=None,
                        help="also write a Surfer ASCII grid (.grd)")
    parser.add_argument("--shp", action=argparse.BooleanOptionalAction, default=None,
                        help="also write the points as a shapefile")
    parser.add_argument("--png", action=argparse.BooleanOptionalAction, default=None,
                        help="also write a PNG preview")
    parser.add_argument("--compress", action="store_true",
                        help="compress the GeoTIFF with compress.py (see --method, --accuracy)")
    parser.add_argument("--method", choices=["dct", "pred", "sz3"], default="dct",
                        help="compression method used with --compress (default dct)")
    parser.add_argument("--accuracy", type=float,
                        help="accuracy used with --compress (default from config.json)")
    parser.add_argument("--quiet", "-q", action="store_true", help="print only the result lines")
    return parser.parse_args(argv)


def settings_from_args(args, cfg):
    """Config values overridden by the command line options."""
    overrides = {
        "neighbours": args.neighbours,
        "resolution": args.resolution,
        "max_distance": args.max_distance,
        "weights": args.weights,
        "leafsize": args.leafsize,
        "smooth_result": args.smooth,
        "smooth_level": args.smooth_level,
        "r_min_function": args.r_min_function,
        "r_min": args.r_min,
        "z_field_name": args.z_field,
        "geoid_correction": args.geoid,
        "geoid_correction_file": os.path.abspath(args.geoid_file) if args.geoid_file else None,
        "geoid_crs": args.geoid_crs,
        "source_crs": args.source_crs,
        "geoid_max_distance": args.geoid_max_distance,
    }
    if args.las_classes:
        overrides["filter_las"] = True
        overrides["filter_classes"] = [int(v) for v in args.las_classes.split(",") if v.strip()]
    overrides = {key: value for key, value in overrides.items() if value is not None}
    return dataclasses.replace(cfg.interpolation, **overrides)


def selected_outputs(args, icfg):
    defaults = {"csv": icfg.csv_result, "surfer": icfg.surfer_grd,
                "shp": icfg.save_points_shp, "png": icfg.preview_png}
    chosen = {"csv": args.csv, "surfer": args.surfer, "shp": args.shp, "png": args.png}
    return {name for name in EXTRA_OUTPUTS if (chosen[name] if chosen[name] is not None else defaults[name])}


def collect_inputs(path):
    if os.path.isfile(path):
        return [path]
    found = []
    for folder, _dirs, names in os.walk(path):
        found.extend(os.path.join(folder, name) for name in names if name.lower().endswith(POINT_EXTENSIONS))
    return sorted(found)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    icfg = settings_from_args(args, cfg)
    outputs = selected_outputs(args, icfg)
    accuracy = args.accuracy if args.accuracy is not None else cfg.compression.accuracy
    compress = (args.method, accuracy) if args.compress else None

    inputs = collect_inputs(args.input)
    if not inputs:
        print(f"Error: no point files ({', '.join(POINT_EXTENSIONS)}) found in {args.input}")
        return 1
    root = os.path.abspath(args.input) if os.path.isdir(args.input) else None

    def log(message):
        if not args.quiet:
            print(message, flush=True)

    failures = 0
    for index, path in enumerate(inputs, 1):
        if root is None:
            out_dir = args.output
        else:
            rel = os.path.relpath(os.path.dirname(os.path.abspath(path)), root)
            out_dir = args.output if rel == "." else os.path.join(args.output, rel)
        log(f"[{index}/{len(inputs)}] {path}")
        try:
            result = run_interpolation(path, out_dir, icfg, outputs, compress, log=log)
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"Error: {os.path.basename(path)}: {exc}", flush=True)
            failures += 1
            continue
        log(f"Interpolated: {result['geotiff']}")
        if "validity" in result:
            print(f"VALIDITY:{result['validity']}", flush=True)
        print(f"Output file: {result.get('compressed', result['geotiff'])}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
