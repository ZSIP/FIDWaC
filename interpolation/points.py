"""Point readers (TXT/CSV, LAS/LAZ, SHP) and the geoid correction, ported from ZSIP/FIDWaC (MIT License)."""

import functools
import io
import os
from typing import Optional, Sequence

import numpy as np
from scipy import spatial

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLED_GEOID = "interpolation/geoid/PL_geod.7z"  # relative to the project folder
POINT_EXTENSIONS = (".txt", ".csv", ".las", ".laz", ".shp")
NODATA = -9999.0


def read_points(path: str, z_field: str = "z", las_classes: Optional[Sequence[int]] = None) -> np.ndarray:
    """Return an (n, 3) array of x, y, z. Rows with NaN or -9999 values are removed, as in ZSIP/FIDWaC."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".csv"):
        xyz = _read_text(path)
    elif ext in (".las", ".laz"):
        xyz = _read_las(path, las_classes)
    elif ext == ".shp":
        xyz = _read_shapefile(path, z_field)
    else:
        raise ValueError(f"Unsupported point file: {path} (supported: {', '.join(POINT_EXTENSIONS)})")
    bad = ~np.isfinite(xyz).all(axis=1) | (xyz == NODATA).any(axis=1)
    return xyz[~bad]


def _read_text(path: str) -> np.ndarray:
    """Comma separated text with one header line; columns 1-3 are x, y, z."""
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1, 2), ndmin=2)
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Cannot read x, y, z columns from {path}: {exc}") from exc
    return data.astype(float)


def _read_las(path: str, las_classes: Optional[Sequence[int]]) -> np.ndarray:
    import laspy

    try:
        las = laspy.read(path)
    except Exception as exc:  # e.g. LAZ without the lazrs package
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    x = np.asarray(las.x, dtype=float)
    y = np.asarray(las.y, dtype=float)
    z = np.asarray(las.z, dtype=float)
    if las_classes:
        keep = np.isin(np.asarray(las.classification), list(las_classes))
        x, y, z = x[keep], y[keep], z[keep]
    return np.column_stack([x, y, z])


def _read_shapefile(path: str, z_field: str) -> np.ndarray:
    import shapefile

    reader = shapefile.Reader(path)
    names = [field[0] for field in reader.fields[1:]]  # the first field is the deletion flag
    if z_field not in names:
        raise ValueError(f"{path}: attribute '{z_field}' not found (available: {', '.join(names)})")
    use_attribute_xy = {"x", "y"}.issubset(names)
    rows = []
    for shape_record in reader.iterShapeRecords():
        attributes = dict(zip(names, shape_record.record))
        z = float(attributes[z_field])
        if use_attribute_xy:
            rows.append((float(attributes["x"]), float(attributes["y"]), z))
        else:
            for px, py in shape_record.shape.points:
                rows.append((px, py, z))
    return np.array(rows, dtype=float).reshape(-1, 3)


def resolve_geoid_path(path: str) -> str:
    """Relative paths (as stored in config.json) are relative to the project folder."""
    if not path:
        raise ValueError("No geoid file configured (geoid_correction_file)")
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_geoid_grid(path: str) -> np.ndarray:
    """Geoid undulations as an (n, 3) array of x, y, z.

    Accepts a .7z archive (its first .csv or .txt member is read in memory) or a text file
    with x, y, z per line, whitespace separated.
    """
    full = resolve_geoid_path(path)
    if not os.path.isfile(full):
        raise ValueError(f"Geoid file not found: {full}")
    return _load_geoid_cached(full, os.path.getmtime(full))  # mtime is part of the cache key


@functools.lru_cache(maxsize=2)
def _load_geoid_cached(path: str, mtime: float) -> np.ndarray:
    if path.lower().endswith(".7z"):
        import py7zr
        from py7zr.io import BytesIOFactory

        with py7zr.SevenZipFile(path, "r") as archive:
            members = [name for name in archive.getnames() if name.lower().endswith((".csv", ".txt"))]
            if not members:
                raise ValueError(f"No .csv or .txt geoid table inside {path}")
            factory = BytesIOFactory(limit=512 * 1024 * 1024)
            archive.extract(targets=members[:1], factory=factory)
        data = np.loadtxt(io.BytesIO(factory.get(members[0]).read()), ndmin=2)
    else:
        data = np.loadtxt(path, ndmin=2)
    if data.shape[1] < 3:
        raise ValueError(f"Geoid table {path} needs three columns: x, y, undulation")
    return data[:, :3]


def apply_geoid_correction(xyz: np.ndarray, geoid_file: str, geoid_crs: str, source_crs: str,
                           max_distance: float = 2000.0, leafsize: int = 10):
    """Subtract the geoid undulation of the nearest geoid point (within max_distance) from z.

    The geoid grid comes from load_geoid_grid (the bundled PL_geod.7z by default).
    Returns (corrected array, number of corrected points).
    """
    from rasterio.crs import CRS
    from rasterio.warp import transform

    geoid = load_geoid_grid(geoid_file)
    gx, gy = geoid[:, 0], geoid[:, 1]
    src, dst = CRS.from_user_input(geoid_crs), CRS.from_user_input(source_crs)
    if src != dst:
        gx, gy = transform(src, dst, gx.tolist(), gy.tolist())
        gx, gy = np.asarray(gx, dtype=float), np.asarray(gy, dtype=float)
    tree = spatial.cKDTree(np.column_stack([gx, gy]), leafsize=leafsize)
    distance, index = tree.query(xyz[:, :2], k=1, workers=-1, distance_upper_bound=max_distance)
    found = np.isfinite(distance)
    corrected = xyz.copy()
    corrected[found, 2] -= geoid[index[found], 2]
    return corrected, int(found.sum())
