"""CRS written to the output GeoTIFF, shared by every compression path and the decoder.

The ``crs`` setting (config.json, the GUI field or ``--crs``) decides it:

* ``""`` (default): the CRS of the input raster, or no CRS when the input has none;
* ``"none"``: no CRS, even when the input has one;
* any other value (``epsg:2180``, ``EPSG:4326``, WKT, PROJ string): that CRS.
"""

from typing import Optional

from rasterio.crs import CRS
from rasterio.errors import CRSError

NO_CRS = "none"


def resolve_output_crs(input_crs: Optional[CRS], setting: Optional[str]) -> Optional[CRS]:
    key = (setting or "").strip()
    if key.lower() == NO_CRS:
        return None
    if not key:
        return input_crs
    try:
        return CRS.from_user_input(key)
    except CRSError as exc:
        raise ValueError(f"invalid CRS setting {key!r}: {exc}") from exc


def crs_from_header(value) -> Optional[CRS]:
    """CRS stored in an archive or payload header; None, '' and 'none' all mean no CRS."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in (NO_CRS, "null"):
        return None
    try:
        return CRS.from_string(text)
    except CRSError as exc:
        raise ValueError(f"invalid CRS in archive header {text!r}: {exc}") from exc


def crs_label(crs: Optional[CRS]) -> str:
    """Tag for output file names: EPSG4326, 'custom' without an EPSG code, '' for no CRS."""
    if crs is None:
        return ""
    code = crs.to_epsg()
    return f"EPSG{code}" if code else "custom"
