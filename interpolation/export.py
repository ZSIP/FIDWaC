"""Writers for IDW results: north-up GeoTIFF, CSV, Surfer ASCII grid, shapefile of points and PNG preview."""

import numpy as np

from interpolation.idw import Grid
from interpolation.points import NODATA


def _clean(values: np.ndarray) -> np.ndarray:
    """NaN and values below -9995 become the NoData value -9999, as in ZSIP/FIDWaC."""
    out = np.asarray(values, dtype=float).copy()
    out[~np.isfinite(out) | (out < -9995)] = NODATA
    return out


def write_geotiff(path: str, values: np.ndarray, grid: Grid, crs: str) -> None:
    """Float32 GeoTIFF with north-up orientation (negative Y pixel size), NoData -9999, DEFLATE."""
    import rasterio
    from rasterio.transform import Affine

    rows = _clean(values)[::-1]  # the last grid row is the northernmost
    transform = Affine(grid.resolution, 0.0, grid.x0, 0.0, -grid.resolution, grid.y0 + grid.ny * grid.resolution)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=grid.ny,
        width=grid.nx,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=NODATA,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(rows.astype(np.float32), 1)


def write_csv(path: str, raw_values: np.ndarray, grid: Grid) -> None:
    """x, y, value for every cell centre; value is the IDW before smoothing, NaN where no point is in range."""
    xs, ys = grid.centres()
    xx, yy = np.meshgrid(xs, ys)
    table = np.column_stack([xx.ravel(), yy.ravel(), np.asarray(raw_values, dtype=float).ravel()])
    np.savetxt(path, table, delimiter=",", header="x,y,value", comments="", fmt="%.6f")


def write_surfer(path: str, values: np.ndarray, grid: Grid) -> None:
    """Surfer ASCII grid (DSAA). Rows run from the south, as Surfer expects."""
    rows = _clean(values)
    valid = rows[rows != NODATA]
    zmin, zmax = (float(valid.min()), float(valid.max())) if valid.size else (NODATA, NODATA)
    xs, ys = grid.centres()
    header = (
        "DSAA\n"
        f"{grid.nx} {grid.ny}\n"
        f"{xs[0]:.6f} {xs[-1]:.6f}\n"
        f"{ys[0]:.6f} {ys[-1]:.6f}\n"
        f"{zmin:.6f} {zmax:.6f}\n"
    )
    with open(path, "w", encoding="ascii") as handle:
        handle.write(header)
        np.savetxt(handle, rows, fmt="%f", delimiter=" ")


def write_points_shapefile(shp_path: str, xyz: np.ndarray, crs: str) -> None:
    """Points with x, y, z attributes. shp_path must end with .shp; a .prj file with the same name records the CRS."""
    import os

    import shapefile
    from rasterio.crs import CRS

    writer = shapefile.Writer(shp_path, shapeType=shapefile.POINT)
    writer.field("x", "F", decimal=10)
    writer.field("y", "F", decimal=10)
    writer.field("z", "F", decimal=10)
    for x, y, z in xyz:
        writer.point(float(x), float(y))
        writer.record(float(x), float(y), float(z))
    writer.close()
    with open(os.path.splitext(shp_path)[0] + ".prj", "w", encoding="utf-8") as handle:
        handle.write(CRS.from_user_input(crs).to_wkt())


def write_preview(path: str, values: np.ndarray, title: str) -> None:
    """Colour preview of the grid (north up), downsampled to about 2000 pixels on the long side."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    rows = np.where(np.isfinite(values) & (values > -9995), values, np.nan)[::-1]
    step = max(1, max(rows.shape) // 2000)
    rows = rows[::step, ::step]
    height, width = rows.shape
    fig = Figure(figsize=(8, max(2.0, 8.0 * height / max(width, 1))), dpi=150)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    image = ax.imshow(rows, cmap="terrain", interpolation="nearest")
    fig.colorbar(image, ax=ax, shrink=0.8, label="z")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(path)
