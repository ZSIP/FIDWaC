#!/usr/bin/env python3
"""Environment and smoke-test verifier for FIDWAC v2."""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # install/ is one level below the project root
sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_MODULES = ["numpy", "scipy", "rasterio", "msgpack", "py7zr", "tqdm", "numba"]
OPTIONAL_MODULES = {
    "flask": "required for the web GUI (app.py)",
}


def _print_result(status: str, label: str, detail: str = "") -> None:
    line = f"[{status}] {label}"
    if detail:
        line = f"{line}: {detail}"
    print(line)


def _import_module(name: str):
    try:
        module = importlib.import_module(name)
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as exc:
        return None, str(exc)
    from importlib import metadata as importlib_metadata

    try:
        version = importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        version = getattr(module, "__version__", None)
    return module, version or "imported"


def check_imports(require_web: bool) -> bool:
    ok = True

    print("== Python package imports ==")
    for name in REQUIRED_MODULES:
        module, detail = _import_module(name)
        if module is None:
            _print_result("FAIL", name, detail)
            ok = False
        else:
            _print_result("OK", name, str(detail))

    for name, purpose in OPTIONAL_MODULES.items():
        module, detail = _import_module(name)
        if module is None:
            _print_result("FAIL" if require_web else "WARN", name, f"{detail} ({purpose})")
            ok = ok and not require_web
        else:
            _print_result("OK", name, str(detail))

    return ok


def check_cuda() -> bool:
    """Report the GPU batch backend (CUDA libraries built by engine/gpu_u8/build.sh) and the device.

    The GPU path does not use numba.cuda: it loads the CUDA libraries through ctypes, so this check
    asks that backend directly. Missing GPU is reported as a warning; the CPU path stays valid."""
    print("\n== CUDA / GPU ==")
    import shutil
    nvcc = shutil.which("nvcc")
    if nvcc:
        _print_result("OK", "nvcc", nvcc)
    else:
        _print_result("WARN", "nvcc", "not on PATH; build engine/gpu_u8/build.sh after installing the CUDA toolkit")
    try:
        from engine.gpu_u8 import phases
    except Exception as exc:  # pragma: no cover - environment-dependent
        _print_result("WARN", "gpu backend", f"not importable ({exc}); CPU path is used")
        return True
    if phases.available():
        _print_result("OK", "gpu backend", "CUDA libraries loaded and a device answers (GPU batch path active)")
    else:
        _print_result("WARN", "gpu backend", f"not active ({phases.reason() or 'libraries not built'}); CPU path is used")
    return True


def check_sz3(require_sz3: bool) -> bool:
    """Round-trip a small array through the pysz binding and report which build is in use."""
    from importlib import metadata

    print("== SZ3 backend (cm=8) ==")
    try:
        import numpy as np
        import pysz
        from pysz import sz, szConfig, szErrorBoundMode
    except (ImportError, OSError, RuntimeError) as exc:
        ok = not require_sz3
        _print_result("FAIL" if require_sz3 else "WARN", "pysz", f"{exc} (needed for --method sz3)")
        return ok

    bound = 0.05
    data = np.linspace(0.0, 10.0, 4096).reshape(64, 64)
    cfg = szConfig()
    cfg.errorBoundMode = szErrorBoundMode.ABS
    cfg.absErrorBound = bound
    comp, _ = sz.compress(data, cfg)
    rec, _ = sz.decompress(comp, np.float64, data.shape)
    err = float(np.max(np.abs(rec - data)))
    ok = err <= bound
    version = metadata.version("pysz")
    _print_result("OK" if ok else "FAIL", "pysz", f"version {version}, round-trip max error {err:.5f} <= {bound}")
    _print_result("OK", "pysz location", str(Path(pysz.__file__).resolve().parent))
    return ok


def check_turbojpeg(require_turbojpeg: bool) -> bool:
    """Encode and decode a small RGB tile with TurboJPEG (cm=1) to catch libjpeg-turbo mismatches."""
    print("== TurboJPEG (cm=1) ==")
    try:
        import numpy as np
        from turbojpeg import TJPF_RGB
        from compress.turbojpeg_loader import open_turbojpeg

        tile = np.random.default_rng(0).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        tj = open_turbojpeg()
        decoded = tj.decode(tj.encode(tile, quality=90, pixel_format=TJPF_RGB), pixel_format=TJPF_RGB)
    except Exception as exc:  # any failure here means JPEG quality mode cannot run
        reason = (str(exc).splitlines() or [type(exc).__name__])[0]
        _print_result(
            "FAIL" if require_turbojpeg else "WARN",
            "turbojpeg",
            f"{reason} (needed for JPEG quality mode, cm=1)",
        )
        return not require_turbojpeg

    ok = decoded.shape == tile.shape
    _print_result("OK" if ok else "FAIL", "turbojpeg", f"encode/decode round-trip, shape {decoded.shape}")
    return ok


def check_archive_tools() -> None:
    print("\n== Archive backend ==")
    tool = shutil.which("7zz")
    if tool:
        _print_result("OK", "7zz (7-Zip-zstd)", tool)
        return
    fallback = next((candidate for candidate in ("7zip", "7z") if shutil.which(candidate)), None)
    if fallback:
        _print_result("WARN", "7zz (7-Zip-zstd)", f"not found; {fallback} is used (run install/setup_linux.sh)")
    else:
        _print_result("WARN", "7zz (7-Zip-zstd)", "not found; py7zr fallback is used (run install/setup_linux.sh)")


INTERPOLATION_MODULES = ("laspy", "lazrs", "shapefile", "matplotlib")


def check_interpolation() -> None:
    """Interpolation (IDW) is optional, so every problem here is a warning, never a failure."""
    print("\n== Interpolation (IDW) ==")
    missing = [name for name in INTERPOLATION_MODULES if _import_module(name)[0] is None]
    if missing:
        _print_result("WARN", "interpolation packages", f"missing: {', '.join(missing)} (run install/setup_linux.sh)")
        return
    _print_result("OK", "interpolation packages", ", ".join(INTERPOLATION_MODULES))
    try:
        import numpy as np

        from interpolation.idw import interpolate_grid, make_grid
        from interpolation.points import BUNDLED_GEOID, load_geoid_grid

        geoid = load_geoid_grid(BUNDLED_GEOID)
        _print_result("OK", "bundled geoid model", f"{len(geoid)} grid points")
        xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        values = interpolate_grid(xy, np.array([1.0, 2.0, 3.0, 4.0]), make_grid(xy, 0.5),
                                  neighbours=4, max_distance=5.0, weights=2.0)
        if np.isfinite(values).any():
            _print_result("OK", "IDW smoke test", f"{values.shape[0]}x{values.shape[1]} grid")
        else:
            _print_result("WARN", "IDW smoke test", "no finite values were produced")
    except Exception as exc:
        _print_result("WARN", "interpolation smoke test", str(exc))


def run_lookup_smoke() -> bool:
    print("\n== Lookup-grid smoke test ==")
    grid_path = PROJECT_ROOT / "models" / "lookup_uint8_grid.npz"
    if not grid_path.exists():
        _print_result("WARN", "lookup_uint8_grid.npz", "file not found; skipping lookup test")
        return True

    try:
        import numpy as np
        predictor_uint8 = importlib.import_module("predictor.predictor_uint8")
        load_uint8_lookup_grid = predictor_uint8.load_uint8_lookup_grid
        predict_uint8_parameters = predictor_uint8.predict_uint8_parameters
        predict_uint8_parameters_batch = predictor_uint8.predict_uint8_parameters_batch
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as exc:
        _print_result("FAIL", "lookup imports", str(exc))
        return False

    grid = load_uint8_lookup_grid(str(grid_path))
    if grid is None:
        _print_result("FAIL", "lookup grid", "failed to load grid data")
        return False

    single_sf, single_bs = predict_uint8_parameters(10.0, 0.5, grid)
    batch_sf, batch_bs = predict_uint8_parameters_batch(
        np.array([5.0, 10.0, 20.0], dtype=np.float32),
        np.array([0.1, 0.5, 0.8], dtype=np.float32),
        grid,
    )
    if single_sf <= 0 or single_bs <= 0 or batch_sf.size != 3 or batch_bs.size != 3:
        _print_result("FAIL", "lookup predictions", "invalid prediction outputs")
        return False

    _print_result(
        "OK",
        "lookup predictions",
        f"single=(sf={single_sf}, bs={single_bs}), batch_bs={sorted(set(batch_bs.tolist()))}",
    )
    return True


def run_roundtrip_smoke(accuracy: float, keep_temp: bool) -> bool:
    print("\n== Compression round-trip smoke test ==")

    try:
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        config_module = importlib.import_module("config")
        compression_module = importlib.import_module("compress.compression")
        decompression_module = importlib.import_module("compress.decompression")
        Config = config_module.Config
        compress_image = compression_module.compress_image
        decompress_file = decompression_module.decompress_file
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as exc:
        _print_result("FAIL", "smoke-test imports", str(exc))
        return False

    temp_dir = Path(tempfile.mkdtemp(prefix="fidwac-smoke-"))
    try:
        input_path = temp_dir / "synthetic_float32.tif"
        output_dir = temp_dir / "out"
        output_dir.mkdir(parents=True, exist_ok=True)

        rows, cols = 32, 32
        yy, xx = np.mgrid[0:rows, 0:cols]
        yy = yy.astype(np.float32)
        xx = xx.astype(np.float32)
        data = ((xx * 0.2) + (yy * 0.1) + np.sin(xx / 4.0) + np.cos(yy / 5.0)).astype(np.float32)

        with rasterio.open(
            input_path,
            "w",
            driver="GTiff",
            height=rows,
            width=cols,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(10.0, 50.0, 1.0, 1.0),
        ) as dst:
            dst.write(data, 1)

        config = Config()
        config.compression.accuracy = accuracy
        config.compression.block_size = 8
        config.compression.auto_select_block_size = False
        config.compression.decimal_places = 2
        config.compression.crs = ""  # empty = keep the input CRS (the check below verifies it)
        config.compression.uint8_accuracy_mode = False
        config.compression.lossless = False
        config.model.backend = "binary"
        config.model.advanced_heuristic = False
        config.model.accept_prediction_if_within_accuracy = False
        config.output.quiet = True

        archive_path = Path(
            compress_image(str(input_path), config=config, num_processes=1, output_dir=str(output_dir))
        )
        restored_path = Path(decompress_file(str(archive_path), config=config, output_dir=str(output_dir)))

        with rasterio.open(input_path) as src:
            original = src.read(1)
            original_transform = src.transform
            original_crs = src.crs

        with rasterio.open(restored_path) as src:
            restored = src.read(1)
            restored_transform = src.transform
            restored_crs = src.crs

        max_error = float(np.max(np.abs(restored.astype(np.float64) - original.astype(np.float64))))
        mean_error = float(np.mean(np.abs(restored.astype(np.float64) - original.astype(np.float64))))
        same_shape = restored.shape == original.shape
        same_transform = restored_transform == original_transform
        same_crs = restored_crs == original_crs

        if not same_shape or not same_transform or not same_crs:
            _print_result(
                "FAIL",
                "round-trip metadata",
                f"shape={same_shape}, transform={same_transform}, crs={same_crs}",
            )
            return False

        if max_error > accuracy + 1e-6:
            _print_result(
                "FAIL",
                "round-trip accuracy",
                f"max_error={max_error:.6f} exceeds requested accuracy={accuracy}",
            )
            return False

        _print_result(
            "OK",
            "round-trip accuracy",
            f"max_error={max_error:.6f}, mean_error={mean_error:.6f}, archive={archive_path.name}",
        )
        if keep_temp:
            _print_result("OK", "temporary files kept", str(temp_dir))
        return True
    finally:
        if not keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify FIDWAC v2 environment and run smoke tests.")
    parser.add_argument("--accuracy", type=float, default=0.05, help="Accuracy used in the round-trip smoke test.")
    parser.add_argument("--skip-roundtrip", action="store_true", help="Skip the compression/decompression smoke test.")
    parser.add_argument("--skip-lookup", action="store_true", help="Skip the uint8 lookup-grid smoke test.")
    parser.add_argument("--require-web", action="store_true", help="Treat a missing Flask (web GUI) as an error.")
    parser.add_argument(
        "--require-turbojpeg",
        action="store_true",
        help="Treat missing TurboJPEG support as an error.",
    )
    parser.add_argument(
        "--require-sz3",
        action="store_true",
        help="Treat a missing or failing SZ3 backend (pysz, --method sz3) as an error.",
    )
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary smoke-test files for inspection.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ok = check_imports(args.require_web)
    ok = check_cuda() and ok
    ok = check_turbojpeg(args.require_turbojpeg) and ok
    ok = check_sz3(args.require_sz3) and ok
    check_archive_tools()
    check_interpolation()

    if not args.skip_lookup:
        ok = run_lookup_smoke() and ok
    if not args.skip_roundtrip:
        ok = run_roundtrip_smoke(args.accuracy, args.keep_temp) and ok

    print("\n== Summary ==")
    if ok:
        _print_result("OK", "environment", "FIDWAC v2 verification passed")
        return 0

    _print_result("FAIL", "environment", "FIDWAC v2 verification failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
