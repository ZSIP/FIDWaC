"""
FIDWAC v2 GUI — reusable widgets
==================================
  _CanvasProgressBar  — flat canvas progress bar
  _StreamToQueue      — stdout/stderr redirect to GUI queue
  ToolTip             — lightweight hover tooltip
  TOOLTIPS            — tooltip text registry
  _tip()              — attach tooltip to any widget
"""

import re as _re
import sys
import tkinter as tk


ANSI_RE = _re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
TQDM_RE = _re.compile(r"\d+%\|")
PCT_RE = _re.compile(r"(\d+)%")



class _CanvasProgressBar:
    """Flat canvas progress bar — guaranteed visible on all platforms.

    Call .set(value) with 0-100 (thread-safe via root.after).
    """

    _COLOR_FILL = "#1565C0"
    _COLOR_BG = "#E3F2FD"
    _COLOR_TXT = "#ffffff"
    _COLOR_TXT2 = "#444444"

    def __init__(self, parent, height: int = 22, **kw):
        kw.setdefault("highlightthickness", 1)
        kw.setdefault("highlightbackground", "#bdbdbd")
        kw.setdefault("relief", "flat")
        self._canvas = tk.Canvas(parent, height=height, bg=self._COLOR_BG, **kw)
        self._pct = 0
        self._canvas.bind("<Configure>", lambda _e: self._draw())

    def pack(self, **kw):
        self._canvas.pack(**kw)

    def grid(self, **kw):
        self._canvas.grid(**kw)

    def set(self, pct: int):
        self._pct = max(0, min(100, int(pct)))
        self._draw()

    def reset(self):
        self._pct = 0
        self._draw()

    def _draw(self):
        self._canvas.delete("all")
        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        if w <= 0 or h <= 0:
            return
        fill_w = int(w * self._pct / 100)
        if fill_w > 0:
            self._canvas.create_rectangle(0, 0, fill_w, h, fill=self._COLOR_FILL, outline="")
        text_col = self._COLOR_TXT if fill_w > w // 2 else self._COLOR_TXT2
        self._canvas.create_text(
            w // 2, h // 2, text=f"{self._pct}%", fill=text_col, font=("TkDefaultFont", 8, "bold")
        )



class _StreamToQueue:
    """Redirect writes to a queue for the GUI log panel; filter tqdm bars.

    tqdm progress lines (matching '72%|████|…') update only the progress
    bar and are not shown in the log.
    The original terminal stream always receives the raw, unmodified output.
    """

    def __init__(self, log_queue, progress_callback=None, original_stream=None):
        self._queue = log_queue
        self._progress_cb = progress_callback
        self._orig = original_stream or sys.stdout
        self._buf = ""

    def _dispatch_line(self, line):
        line = line.strip()
        if not line:
            return
        line = ANSI_RE.sub("", line)
        if TQDM_RE.search(line):
            m = PCT_RE.search(line)
            if m and self._progress_cb:
                self._progress_cb(int(m.group(1)), line)
        else:
            self._queue.put(line)

    def _update_progress_from_partial_buffer(self):
        if not self._progress_cb:
            return
        line = self._buf.rsplit("\r", 1)[-1].rsplit("\n", 1)[-1].strip()
        line = ANSI_RE.sub("", line)
        m = PCT_RE.search(line)
        if m:
            self._progress_cb(int(m.group(1)), line)

    def write(self, text):
        if not text:
            return
        try:
            self._orig.write(text)
            self._orig.flush()
        except Exception:
            pass

        self._buf += text
        self._update_progress_from_partial_buffer()

        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if "\r" in line:
                line = line.rsplit("\r", 1)[-1]
            self._dispatch_line(line)

    def flush(self):
        line = self._buf.strip()
        self._buf = ""
        self._dispatch_line(line)

    def fileno(self):
        try:
            return self._orig.fileno()
        except Exception:
            return 1

    def isatty(self):
        try:
            return self._orig.isatty()
        except Exception:
            return True

    @property
    def encoding(self):
        return getattr(self._orig, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._orig, "errors", "replace")

    def writable(self):
        return True


# Log tag support - send (text, tag) tuples for colored logs


def setup_log_text_colors(text_widget):
    """Configure Text widget color tags for log output."""
    text_widget.tag_configure("INFO", foreground="#89c3e8")
    text_widget.tag_configure("SUCCESS", foreground="#4ec94e")
    text_widget.tag_configure("ERROR", foreground="#ff6b6b")
    text_widget.tag_configure("WARNING", foreground="#ffa966")
    text_widget.tag_configure("DEBUG", foreground="#a8a8a8")
    text_widget.tag_configure("FILE", foreground="#b4b4ff")
    text_widget.tag_configure("STATS", foreground="#ffdb58", font=("TkFixedFont", 8, "bold"))


def detect_log_tag(line: str) -> str:
    """Auto-detect log message type and return appropriate tag."""
    line_lower = line.lower()
    result = ""
    if any(x in line_lower for x in ["error", "failed", "exception", "traceback"]):
        result = "ERROR"
    elif any(x in line_lower for x in ["success", "done", "completed", "✓"]):
        result = "SUCCESS"
    elif any(x in line_lower for x in ["warning", "warn", "deprecated"]):
        result = "WARNING"
    elif any(x in line_lower for x in ["debug", "verbose"]):
        result = "DEBUG"
    elif any(x in line_lower for x in ["→", "output:", "processing", "compressing", "decompressing"]):
        result = "FILE"
    elif any(x in line_lower for x in ["found", "files", "total", "ratio", "size"]):
        result = "STATS"
    return result



class ToolTip:
    """Lightweight tooltip for any tkinter widget."""

    def __init__(self, widget, text: str, delay_ms: int = 800):
        self._widget = widget
        self._text = text
        self._tip = None
        self._delay_ms = delay_ms  # Delay before showing tooltip (ms)
        self._timer = None  # Timer ID for scheduling
        widget.bind("<Enter>", self._on_enter)
        widget.bind("<Leave>", self._on_leave)
        widget.bind("<Motion>", self._on_motion)

    def _on_enter(self, _event=None):
        """Start timer when mouse enters widget."""
        if self._timer:
            self._widget.after_cancel(self._timer)
        self._timer = self._widget.after(self._delay_ms, self._show)

    def _on_leave(self, _event=None):
        """Cancel timer and hide tooltip when mouse leaves."""
        if self._timer:
            self._widget.after_cancel(self._timer)
            self._timer = None
        self._hide()

    def _on_motion(self, _event=None):
        """Reset timer on mouse motion (keeps tooltip hidden if moving)."""
        if self._timer:
            self._widget.after_cancel(self._timer)
            self._timer = self._widget.after(self._delay_ms, self._show)

    def _show(self, _event=None):
        if self._tip or not self._text:
            return
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tip = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tw,
            text=self._text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("TkDefaultFont", 8),
            wraplength=320,
        ).pack(ipadx=5, ipady=3)

    def _hide(self, _event=None):
        if self._tip:
            self._tip.destroy()
            self._tip = None


def _tip(widget, key: str):
    """Attach tooltip by key and return widget (for chaining)."""
    if key in TOOLTIPS:
        ToolTip(widget, TOOLTIPS[key])
    return widget


TOOLTIPS = {
    "input_type_file": "Compress a single file (GeoTIFF or raw float data).",
    "input_type_directory": "Recursively compress all files in directory.",
    "input_path": "Path to input file or directory.",
    "browse_input": "Browse file system to select input.",
    "output_dir": "Output directory for compressed/decompressed files.",
    "browse_output": "Browse file system to select output directory.",
    "rgb_channel_indices": "Map RGB channels to YCbCr indices\n"
    "(1=Y(luma), 2=Cb, 3=Cr). Leave 0 to disable channel.",
    "run_button": "Start compression/decompression with current settings.",
    "stop_button": "Stop running compression process.",
    "clear_log": "Clear log output.",
    "accuracy": "Maximum reconstruction error per pixel (float data).\n"
    "0.01 = high quality | 0.5 = high compression\n"
    "8-bit input takes its accuracy from the 8-bit section.",
    "method": "Compression path:\n"
    "dct = classic DCT workflow (default)\n"
    "pred = near-lossless predictive cm=7 path\n"
    "sz3 = external SZ3 absolute-error path (cm=8)\n"
    "pred and sz3 code every band separately (RGB and multi-band too).\n"
    "Keep dct/pred/sz3 as distinct modes; choose based on error target and file size.",
    "block_size": "DCT block size in pixels.\n"
    "Larger blocks → better compression for smooth terrain.\n"
    "Smaller blocks → better for complex/varied data.",
    "auto_select": "Test block sizes 8/16/32 on a sample and pick\n"
    "the one producing the smallest output file.",
    "backend": "heuristic – rule-based prediction with lookup tables\n"
    "binary    – binary search only (no prediction)",
    "processes": "Number of parallel worker processes.\n" "More = faster on multi-core CPUs.",
    "gpu": "auto: use the GPU when it is found, otherwise the CPU (with a note).\n"
           "on: require the GPU. off: CPU only. The number of processes that get a GPU context\n"
           "is measured on this machine.",
    "files_parallel": "Number of files to process simultaneously.\n"
    "1 = single file gets all processes (max compression speed)\n"
    ">1 = multiple files share processes (better I/O, more parallelism)",
    "crs": "CRS of the output GeoTIFF.\n"
    "Empty = CRS of the input file (none if the input has no CRS).\n"
    "An EPSG code (e.g. epsg:2180) replaces the input CRS.",
    "decimal_places": "DCT coefficient rounding precision.\n" "2 → round to 0.01 step.",
    "auto_sample_size": "How many blocks to compress during auto-select test.\n"
    "Higher = more accurate estimate, but slower start.",
    "std_high": "If image std > this → only N=8 blocks allowed\n" "(very heterogeneous terrain).",
    "std_medium": "If image std > this → N=8 or N=16 only\n" "(moderately varied terrain).",
    "advanced_heuristic": "Use 16-feature lookup table for prediction\n"
    "instead of simple std-based heuristic.",
    "minimize_backscan": "Backscan steps after binary search to minimize L.\n"
    "Active for ALL backends (binary, heuristic).\n"
    "Higher = smaller output file, more computation.",
    "backscan_break": "Stop backscan after this many consecutive failures.\n"
    "Active for ALL backends.\n"
    "0 = exhaustive (smallest file), >0 = faster but larger file.",
    "accept_prediction": "Accept prediction directly when reconstruction error is within\n"
    "accuracy threshold (float & uint8). Skips binary search — faster.\n"
    "Works with both prediction backends.",
    "blocks_per_process": "Number of blocks sent to each worker in one batch.\n"
    "Larger batches reduce IPC overhead.",
    "fast_eval": "Use precomputed IDCT basis matrices for faster\n" "block error evaluation.",
    "full_idct": "Validate final block error using full IDCT\n" "reconstruction (more accurate).",
    "l2_precheck": "Reject obviously bad candidates early using L2 norm\n"
    "before running expensive full reconstruction.",
    "incremental": "Incremental backscan with early termination\n" "when improvement stalls.",
    "compression_method": "7z compression algorithm.\n"
    "LZMA2  – multi-threaded, balanced\n"
    "PPMD   – best ratio, single-threaded\n"
    "BZIP2  – medium speed/ratio\n"
    "DEFLATE – fastest, least compression",
    "lzma2_block_size": "LZMA2 block (chunk) size of the .7z archive, e.g. 1m, 2m, 4m.\n"
    "7-Zip compresses the blocks in parallel, so smaller blocks\n"
    "are much faster with nearly the same size.\n"
    "Empty = 7-Zip default. Only used with LZMA2.",
    "no_crs": "Write the output GeoTIFF without a CRS, even if the input has one\n"
    "(e.g. Mars DEM, synthetic grids, non-Earth rasters).",
    "crs_epsg": "CRS of the output GeoTIFF. Empty = CRS of the input file.\n"
    "Example: epsg:2180 (Poland PUWG 1992)\n"
    "         epsg:4326 (WGS84 geographic)\n"
    "         epsg:3857 (Web Mercator)",
    "tiff_compression": "Internal compression of the output GeoTIFF file.",
    "delete_temp": "Delete intermediate temporary files after each step.",
    "quiet": "Suppress all progress messages in the log panel.",
    "show_log_panel": "Show or hide the log panel at the bottom of the window.\n"
    "Hide to save space, show to monitor compression progress.",
    "rgb_quality": "JPEG quality % for 8-bit data in Quality mode (dct; RGB uses JPEG).\n"
    "85% = good quality (default), 100% = visually lossless,\n"
    "50% = medium compression, 25% = high compression.\n"
    "Accuracy is set in the Accuracy field of the 8-bit section, not with this slider.",
    "lossless": "Lossless mode (any data type): the raster is stored unchanged,\n"
    "3-band uint8 RGB as PNG + world file, everything else as GeoTIFF (DEFLATE),\n"
    "packed with the georeferencing into the FIDWAC .7z archive.\n"
    "Only organises the data in the FIDWAC format: little or no size reduction.",
    "uint8_quality": "Fast JPEG-like quantization for 8-bit data.\n"
    "Quality slider controls compression vs quality trade-off.\n"
    "Fastest option, no strict error guarantee.",
    "uint8_accuracy": "Binary search DCT for 8-bit data.\n"
    "Guarantees max color value error <= Accuracy threshold.\n"
    "Data is centered (-128) before DCT for optimal compression.\n"
    "Slower than quality mode, but error-controlled.",
    "uint8_accuracy_px": "Maximum allowed color value error for 8-bit data, in pixel values (0-255).\n"
    "Integer value: 2 = minimum (due to YCbCr rounding), higher = more error allowed.\n"
    "Used by dct Accuracy mode and by pred / sz3 on 8-bit input.\n"
    "Lower = better quality and larger file.",
    "uint8_lossless": "Lossless mode: data stored unchanged (PNG for 3-band RGB,\n"
    "GeoTIFF otherwise) with the georeferencing in .7z.\n"
    "No quality loss and little or no size reduction.",
    "allowed_block_sizes_8": "Allow N=8 block size in auto-select.\n"
    "Smaller blocks → better for complex/varied data.\n"
    "Slower compression but more accurate.",
    "allowed_block_sizes_16": "Allow N=16 block size in auto-select.\n"
    "Balanced option for most terrain types.\n"
    "Good compression ratio with reasonable speed.",
    "allowed_block_sizes_32": "Allow N=32 block size in auto-select.\n"
    "Larger blocks → best compression for smooth terrain.\n"
    "Fastest option, but may not meet accuracy for complex data.",
    "ycbcr_multiplier": "Per-channel accuracy multiplier for YCbCr.\n"
    "1.0 = use full accuracy for this channel.\n"
    "0.9 = allow 10% more error (smaller file).\n"
    "Lower = more compression, higher error tolerance.",
}


__all__ = [
    "_CanvasProgressBar",
    "_StreamToQueue",
    "ToolTip",
    "_tip",
    "TOOLTIPS",
    "ANSI_RE",
    "TQDM_RE",
    "PCT_RE",
    "setup_log_text_colors",
    "detect_log_tag",
]
