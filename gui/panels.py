"""
FIDWAC v2 GUI — panel builders
================================
Mixin class GUIPanels providing:
  _build_main_tab()
  _build_advanced_tab()
  _build_buttons()
  _build_progress()
  _build_log()
  _add_spinbox()
"""

import tkinter as tk
from tkinter import ttk

from .widgets import _CanvasProgressBar, _tip, setup_log_text_colors


class GUIPanels:
    """Mixin class providing GUI panel builders."""

    def __init__(self):
        """Initialize GUIPanels attributes to None."""
        self.input_type = None
        self.input_path = None
        self._op_label = None
        self.output_dir = None
        self._output_label = None
        self.allow_n8 = None
        self.allow_n16 = None
        self.allow_n32 = None
        self.method = None
        self.accuracy = None
        self._acc_text = None
        self._acc_spin = None
        self.block_size = None
        self.auto_select = None
        self._block_size_cb = None
        self.backend = None
        self._backend_cb = None
        self.uint8_mode = None
        self._rq_label_widget = None
        self.rgb_quality = None
        self._rq_scale = None
        self._rq_label = None
        self._u8acc_label_widget = None
        self.uint8_accuracy = None
        self._u8acc_spin = None
        self._u8acc_hint = None
        self._u8rgb_label = None
        self.rgb_ch_r = None
        self.rgb_ch_g = None
        self.rgb_ch_b = None
        self.ycbcr_y_mult = None
        self.ycbcr_cb_mult = None
        self.ycbcr_cr_mult = None
        self.advanced_heuristic = None
        self._adv_heuristic_cb = None
        self.accept_prediction = None
        self._accept_pred_cb = None
        self._minimize_backscan_spin = None
        self._backscan_break_spin = None
        self._auto_sample_spin = None
        self._std_high_spin = None
        self._std_medium_spin = None
        self.source_dir = None
        self.results_dir = None
        self.processes = None
        self.files_parallel = None
        self.compression_method = None
        self.lzma2_block_size = None
        self.tiff_compression = None
        self.crs = None
        self._crs_entry = None
        self.no_crs = None
        self.delete_temp = None
        self.quiet = None
        self.show_log_panel = None
        self._run_button = None
        self._stop_button = None
        self._progress = None
        self._status_label = None
        self._log_frame = None
        self._log_text = None

    # =====================================================================
    # MAIN TAB
    # =====================================================================

    def _build_main_tab(self, parent, cfg):
        # Input and Output, full width
        parent.columnconfigure(0, weight=1)

        # ── INPUT (full width) ──────────────────────────────────────────────
        f = ttk.LabelFrame(parent, text="Input", padding="8", style="Input.TLabelframe")
        f.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 4))
        f.columnconfigure(2, weight=1)

        self.input_type = tk.StringVar(value="file")
        ttk.Label(f, text="Type:").grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
        _tip(
            ttk.Radiobutton(f, text="File", variable=self.input_type, value="file"),
            "input_type_file",
        ).grid(row=0, column=1, sticky=tk.W)
        _tip(
            ttk.Radiobutton(f, text="Directory", variable=self.input_type, value="directory"),
            "input_type_directory",
        ).grid(row=0, column=2, sticky=tk.W)

        ttk.Label(f, text="Path:").grid(row=1, column=0, sticky=tk.W, padx=(0, 4), pady=4)
        self.input_path = tk.StringVar()
        self.input_path.trace_add("write", self._on_input_changed)
        _tip(ttk.Entry(f, textvariable=self.input_path), "input_path").grid(
            row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=4
        )
        _tip(ttk.Button(f, text="Browse…", command=self._browse_input), "browse_input").grid(
            row=1, column=3, padx=(6, 0)
        )
        self._op_label = ttk.Label(
            f, text="", foreground="#005500", font=("TkDefaultFont", 9, "bold")
        )
        self._op_label.grid(row=2, column=0, columnspan=4, sticky=tk.W)

        # ── OUTPUT (full width) ─────────────────────────────────────────────
        f = ttk.LabelFrame(parent, text="Output", padding="8", style="Output.TLabelframe")
        f.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 4))
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="Directory:").grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
        self.output_dir = tk.StringVar()
        self.output_dir.trace_add("write", self._on_output_changed)
        _tip(ttk.Entry(f, textvariable=self.output_dir), "output_dir").grid(
            row=0, column=1, sticky=(tk.W, tk.E), pady=4
        )
        _tip(ttk.Button(f, text="Browse…", command=self._browse_output), "browse_output").grid(
            row=0, column=2, padx=(6, 0)
        )
        self._output_label = ttk.Label(
            f, text="", foreground="#89c3e8", font=("TkDefaultFont", 9, "bold")
        )
        self._output_label.grid(row=1, column=0, columnspan=3, sticky=tk.W)

        # ── COMPRESSION SETTINGS (full width) ───────────────────────────────
        fc = ttk.LabelFrame(
            parent, text="Compression Settings", padding="8", style="Compress.TLabelframe"
        )
        fc.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=4)
        fc.columnconfigure(1, weight=1)

        self._build_float_section(fc, cfg)
        self._build_uint8_section(fc, cfg)

        # Allowed block sizes checkboxes
        self.allow_n8 = tk.BooleanVar(value=8 in cfg.compression.allowed_block_sizes)
        self.allow_n16 = tk.BooleanVar(value=16 in cfg.compression.allowed_block_sizes)
        self.allow_n32 = tk.BooleanVar(value=32 in cfg.compression.allowed_block_sizes)
        bs_frame = ttk.Frame(fc)
        bs_frame.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=3)
        ttk.Label(bs_frame, text="Auto-select N:").pack(side=tk.LEFT)
        _tip(
            ttk.Checkbutton(
                bs_frame, text="8", variable=self.allow_n8, command=self._on_allowed_bs_change
            ),
            "allowed_block_sizes_8",
        ).pack(side=tk.LEFT, padx=2)
        _tip(
            ttk.Checkbutton(
                bs_frame, text="16", variable=self.allow_n16, command=self._on_allowed_bs_change
            ),
            "allowed_block_sizes_16",
        ).pack(side=tk.LEFT, padx=2)
        _tip(
            ttk.Checkbutton(
                bs_frame, text="32", variable=self.allow_n32, command=self._on_allowed_bs_change
            ),
            "allowed_block_sizes_32",
        ).pack(side=tk.LEFT, padx=2)

        self._on_uint8_mode_change()

    def set_accuracy(self, value):
        """Set the float accuracy from code (preset, button); slider and entry follow."""
        self.accuracy.set(value)
        self._acc_text.set(f"{value:.3f}")

    def _follow_typed_accuracy(self):
        """Move the slider while a valid number is being typed; the text itself is left alone."""
        try:
            value = float(self._acc_text.get().replace(",", "."))
        except ValueError:
            return
        if 0.001 <= value <= 1.0 and abs(value - self.accuracy.get()) > 1e-12:
            self.accuracy.set(value)

    def _commit_accuracy(self):
        """Take the typed accuracy; an invalid or out-of-range entry reverts to the last valid value."""
        try:
            value = float(self._acc_text.get().replace(",", "."))
        except ValueError:
            value = None
        if value is None or not 0.001 <= value <= 1.0:
            value = self.accuracy.get()
        self.set_accuracy(value)

    def _build_float_section(self, parent, cfg):
        float_frame = ttk.LabelFrame(
            parent,
            text="Float data  (elevation / DSM / DTM)",
            padding="6",
            style="Float.TLabelframe",
        )
        float_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 4))
        float_frame.columnconfigure(1, weight=1)

        # Accuracy: slider, typed value (spinbox) and quick buttons share one variable
        ttk.Label(float_frame, text="Accuracy (float):").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.accuracy = tk.DoubleVar(value=cfg.compression.accuracy)
        self._acc_text = tk.StringVar(value=f"{cfg.compression.accuracy:.3f}")
        acc_frame = ttk.Frame(float_frame)
        acc_frame.grid(row=0, column=1, sticky=tk.W)
        _tip(
            ttk.Scale(
                acc_frame,
                from_=0.01,
                to=1.0,
                orient=tk.HORIZONTAL,
                variable=self.accuracy,
                length=200,
                command=lambda v: self._acc_text.set(f"{float(v):.3f}"),
            ),
            "accuracy",
        ).grid(row=0, column=0, sticky=tk.W)
        self._acc_spin = _tip(
            ttk.Spinbox(
                acc_frame,
                from_=0.001,
                to=1.0,
                increment=0.01,
                format="%.3f",
                textvariable=self._acc_text,
                width=7,
                command=self._commit_accuracy,
            ),
            "accuracy",
        )
        self._acc_spin.grid(row=0, column=1, sticky=tk.W, padx=4)
        self._acc_spin.bind("<Return>", lambda _e: self._commit_accuracy())
        self._acc_spin.bind("<FocusOut>", lambda _e: self._commit_accuracy())
        self._acc_text.trace_add("write", lambda *_a: self._follow_typed_accuracy())

        # Predefined values on their own row so that they are never clipped
        preset_frame = ttk.Frame(acc_frame)
        preset_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(3, 0))
        for label, val in [("0.01", 0.01), ("0.05", 0.05), ("0.1", 0.1), ("0.5", 0.5), ("1.0", 1.0)]:
            ttk.Button(
                preset_frame,
                text=label,
                width=5,
                command=lambda v=val: self.set_accuracy(v),
            ).pack(side=tk.LEFT, padx=1)

        # Compression path selector
        ttk.Label(float_frame, text="Method:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.method = tk.StringVar(value=getattr(cfg.compression, "method", "dct"))
        method_combo = _tip(
            ttk.Combobox(
                float_frame,
                textvariable=self.method,
                values=["dct", "pred", "sz3"],
                width=12,
                state="readonly",
            ),
            "method",
        )
        method_combo.grid(row=1, column=1, sticky=tk.W)
        method_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_uint8_mode_change())
        ttk.Label(
            float_frame,
            text="dct = standard / pred = cm=7 / sz3 = external cm=8",
            foreground="gray",
            font=("TkDefaultFont", 8),
        ).grid(row=1, column=2, sticky=tk.W, padx=(10, 0))

        # Block size + auto-select
        ttk.Label(float_frame, text="Block size:").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.block_size = tk.StringVar(value=str(cfg.compression.block_size))
        self.auto_select = tk.BooleanVar(value=cfg.compression.auto_select_block_size)
        bs_frame = ttk.Frame(float_frame)
        bs_frame.grid(row=2, column=1, sticky=tk.W)
        self._block_size_cb = _tip(
            ttk.Combobox(
                bs_frame,
                textvariable=self.block_size,
                values=["8", "16", "32"],
                width=6,
                state="readonly",
            ),
            "block_size",
        )
        self._block_size_cb.pack(side=tk.LEFT)
        _tip(
            ttk.Checkbutton(
                bs_frame,
                text="Auto-select",
                variable=self.auto_select,
                command=self._on_auto_select_change,
            ),
            "auto_select",
        ).pack(side=tk.LEFT, padx=10)
        self._on_auto_select_change()

        # Backend
        ttk.Label(float_frame, text="Backend:").grid(row=3, column=0, sticky=tk.W, pady=3)
        self.backend = tk.StringVar(value=cfg.model.backend)
        self._backend_cb = _tip(
            ttk.Combobox(
                float_frame,
                textvariable=self.backend,
                values=["binary", "heuristic"],
                width=12,
                state="readonly",
            ),
            "backend",
        )
        self._backend_cb.grid(row=3, column=1, sticky=tk.W, pady=3)
        self._backend_cb.bind("<<ComboboxSelected>>", self._on_backend_change)

    def _build_uint8_section(self, parent, cfg):
        uint_frame = ttk.LabelFrame(
            parent, text="8-bit data  (RGB / ortho / uint8)", padding="6", style="Uint8.TLabelframe"
        )
        uint_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 4))
        uint_frame.columnconfigure(1, weight=1)

        if cfg.compression.lossless:
            _mode_init = "lossless"
        elif cfg.compression.uint8_accuracy_mode:
            _mode_init = "accuracy"
        else:
            _mode_init = "quality"
        self.uint8_mode = tk.StringVar(value=_mode_init)
        mode_frame = ttk.Frame(uint_frame)
        mode_frame.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT, padx=(0, 6))
        _tip(
            ttk.Radiobutton(
                mode_frame,
                text="Quality",
                variable=self.uint8_mode,
                value="quality",
                command=self._on_uint8_mode_change,
            ),
            "uint8_quality",
        ).pack(side=tk.LEFT)
        _tip(
            ttk.Radiobutton(
                mode_frame,
                text="Accuracy",
                variable=self.uint8_mode,
                value="accuracy",
                command=self._on_uint8_mode_change,
            ),
            "uint8_accuracy",
        ).pack(side=tk.LEFT, padx=(6, 0))
        _tip(
            ttk.Radiobutton(
                mode_frame,
                text="Lossless PNG",
                variable=self.uint8_mode,
                value="lossless",
                command=self._on_uint8_mode_change,
            ),
            "uint8_lossless",
        ).pack(side=tk.LEFT, padx=(6, 0))

        # Quality slider
        self._rq_label_widget = ttk.Label(uint_frame, text="Quality (%):")
        self._rq_label_widget.grid(row=1, column=0, sticky=tk.W, pady=3)
        self.rgb_quality = tk.IntVar(value=cfg.compression.rgb_quality)
        rq_frame = ttk.Frame(uint_frame)
        rq_frame.grid(row=1, column=1, sticky=tk.W)
        self._rq_scale = _tip(
            ttk.Scale(
                rq_frame,
                from_=25,
                to=100,
                variable=self.rgb_quality,
                orient=tk.HORIZONTAL,
                length=120,
                command=lambda v: self._rq_label.config(text=f"{int(float(v))}%"),
            ),
            "rgb_quality",
        )
        self._rq_scale.pack(side=tk.LEFT)
        self._rq_label = ttk.Label(rq_frame, text=f"{cfg.compression.rgb_quality}%", width=4)
        self._rq_label.pack(side=tk.LEFT, padx=4)

        # Accuracy spinbox
        self._u8acc_label_widget = ttk.Label(uint_frame, text="Accuracy (8-bit):")
        self._u8acc_label_widget.grid(row=2, column=0, sticky=tk.W, pady=3)
        self.uint8_accuracy = tk.IntVar(value=cfg.compression.uint8_accuracy)
        u8acc_frame = ttk.Frame(uint_frame)
        u8acc_frame.grid(row=2, column=1, sticky=tk.W)
        self._u8acc_spin = _tip(
            ttk.Spinbox(u8acc_frame, from_=2, to=20, textvariable=self.uint8_accuracy, width=5),
            "uint8_accuracy_px",
        )
        self._u8acc_spin.pack(side=tk.LEFT)
        self._u8acc_hint = ttk.Label(
            uint_frame, text="max color error per pixel  (2 = minimum, due to YCbCr rounding)", foreground="gray", wraplength=350
        )
        self._u8acc_hint.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=(20, 0), pady=(0, 4))

        # RGB channel indices
        self._u8rgb_label = ttk.Label(uint_frame, text="RGB channels:")
        self._u8rgb_label.grid(row=4, column=0, sticky=tk.W, pady=3)
        _rgb_def = list(getattr(cfg.compression, "rgb_channel_indices", [1, 2, 3]))
        if len(_rgb_def) != 3:
            _rgb_def = [1, 2, 3]
        self.rgb_ch_r = tk.IntVar(value=_rgb_def[0])
        self.rgb_ch_g = tk.IntVar(value=_rgb_def[1])
        self.rgb_ch_b = tk.IntVar(value=_rgb_def[2])
        u8rgb_frame = ttk.Frame(uint_frame)
        u8rgb_frame.grid(row=4, column=1, sticky=tk.W)
        for lbl, var in [("R:", self.rgb_ch_r), ("G:", self.rgb_ch_g), ("B:", self.rgb_ch_b)]:
            ttk.Label(u8rgb_frame, text=lbl).pack(side=tk.LEFT)
            _tip(
                ttk.Spinbox(u8rgb_frame, from_=1, to=20, textvariable=var, width=3),
                "rgb_channel_indices",
            ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(u8rgb_frame, text="(YCbCr − smaller; 0 = off)", foreground="gray").pack(
            side=tk.LEFT, padx=6
        )

        # YCbCr multipliers (hidden from GUI, loaded from config)
        self.ycbcr_y_mult = tk.DoubleVar(value=cfg.compression.ycbcr_y_multiplier)
        self.ycbcr_cb_mult = tk.DoubleVar(value=cfg.compression.ycbcr_cb_multiplier)
        self.ycbcr_cr_mult = tk.DoubleVar(value=cfg.compression.ycbcr_cr_multiplier)

    # =====================================================================
    # ADVANCED TAB
    # =====================================================================

    def _build_advanced_tab(self, parent, cfg):
        # ── Two-column layout ─────────────────────────────────────────────
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

        left = ttk.Frame(parent)
        right = ttk.Frame(parent)
        left.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 6))
        right.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N))
        left.columnconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        # ── LEFT: Model / Search ──────────────────────────────────────────
        fm = ttk.LabelFrame(left, text="Model / Search", padding="8", style="Model.TLabelframe")
        fm.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=4)
        fm.columnconfigure(1, weight=1)

        self.advanced_heuristic = tk.BooleanVar(value=cfg.model.advanced_heuristic)
        self._adv_heuristic_cb = _tip(
            ttk.Checkbutton(
                fm, text="Advanced heuristic (lookup table)", variable=self.advanced_heuristic
            ),
            "advanced_heuristic",
        )
        self._adv_heuristic_cb.grid(row=0, column=0, sticky=tk.W, pady=2)

        self.accept_prediction = tk.BooleanVar(value=cfg.model.accept_prediction_if_within_accuracy)
        self._accept_pred_cb = _tip(
            ttk.Checkbutton(
                fm, text="Accept prediction if within accuracy", variable=self.accept_prediction
            ),
            "accept_prediction",
        )
        self._accept_pred_cb.grid(row=1, column=0, sticky=tk.W, pady=2)

        self._minimize_backscan_spin = self._add_spinbox(
            fm,
            2,
            "Minimize backscan:",
            "minimize_backscan",
            cfg.model.minimize_backscan,
            0,
            30,
            "minimize_backscan",
        )
        self._backscan_break_spin = self._add_spinbox(
            fm,
            3,
            "Backscan break after:",
            "backscan_break",
            cfg.model.backscan_break_after,
            0,
            30,
            "backscan_break",
        )

        # ── LEFT: Auto-select ─────────────────────────────────────────────
        fa = ttk.LabelFrame(
            left, text="Auto-select Settings", padding="8", style="Model.TLabelframe"
        )
        fa.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=4)
        fa.columnconfigure(1, weight=1)

        self._auto_sample_spin = self._add_spinbox(
            fa,
            0,
            "Sample size:",
            "auto_sample_size",
            cfg.compression.auto_select_sample_size,
            10,
            10000,
            "auto_sample_size",
        )
        self._std_high_spin = self._add_spinbox(
            fa,
            1,
            "Std threshold high:",
            "std_high",
            cfg.compression.auto_select_std_threshold_high,
            0,
            100,
            "std_high",
        )
        self._std_medium_spin = self._add_spinbox(
            fa,
            2,
            "Std threshold medium:",
            "std_medium",
            cfg.compression.auto_select_std_threshold_medium,
            0,
            100,
            "std_medium",
        )

        # Hidden vars
        self.source_dir = tk.StringVar(value=cfg.source_dir)
        self.results_dir = tk.StringVar(value=cfg.results_dir)

        # ── RIGHT: Parallel Processing ────────────────────────────────────
        fp = ttk.LabelFrame(
            right, text="Parallel Processing", padding="8", style="Compress.TLabelframe"
        )
        fp.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=4)
        fp.columnconfigure(1, weight=1)

        ttk.Label(fp, text="Processes:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.processes = tk.IntVar(value=self._cpu_count)
        proc_frame = ttk.Frame(fp)
        proc_frame.grid(row=0, column=1, sticky=tk.W)
        _tip(
            ttk.Spinbox(
                proc_frame, from_=1, to=self._cpu_count, textvariable=self.processes, width=6
            ),
            "processes",
        ).pack(side=tk.LEFT)
        ttk.Label(proc_frame, text=f"  max {self._cpu_count} cores", foreground="gray").pack(
            side=tk.LEFT
        )

        ttk.Label(fp, text="Files parallel:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.files_parallel = tk.IntVar(value=1)
        ff_frame = ttk.Frame(fp)
        ff_frame.grid(row=1, column=1, sticky=tk.W)
        _tip(
            ttk.Spinbox(
                ff_frame, from_=1, to=self._cpu_count, textvariable=self.files_parallel, width=6
            ),
            "files_parallel",
        ).pack(side=tk.LEFT)
        ttk.Label(ff_frame, text="  (1=single, >1=multi)", foreground="gray").pack(side=tk.LEFT)

        ttk.Label(fp, text="GPU:").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.gpu = tk.StringVar(value="auto")
        gpu_frame = ttk.Frame(fp)
        gpu_frame.grid(row=2, column=1, sticky=tk.W)
        _tip(
            ttk.Combobox(
                gpu_frame, textvariable=self.gpu, values=("auto", "on", "off"), width=6,
                state="readonly",
            ),
            "gpu",
        ).pack(side=tk.LEFT)
        ttk.Label(gpu_frame, text="  auto = GPU if found, else CPU", foreground="gray").pack(
            side=tk.LEFT
        )

        # ── RIGHT: Output Settings ────────────────────────────────────────
        fo = ttk.LabelFrame(right, text="Output Settings", padding="8", style="Perf.TLabelframe")
        fo.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=4)
        fo.columnconfigure(1, weight=1)

        ttk.Label(fo, text="Archive method:").grid(row=0, column=0, sticky=tk.W, pady=2)
        method_default = (
            cfg.output.compression_method
            if cfg.output.compression_method in ["LZMA2", "PPMD", "BZIP2", "DEFLATE"]
            else "PPMD"
        )
        self.compression_method = tk.StringVar(value=method_default)
        _tip(
            ttk.Combobox(
                fo,
                textvariable=self.compression_method,
                values=["LZMA2", "PPMD", "BZIP2", "DEFLATE"],
                width=10,
                state="readonly",
            ),
            "compression_method",
        ).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(fo, text="LZMA2 block size:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.lzma2_block_size = tk.StringVar(value=cfg.output.lzma2_block_size)
        _tip(
            ttk.Combobox(
                fo,
                textvariable=self.lzma2_block_size,
                values=["", "512k", "1m", "2m", "4m", "8m"],
                width=10,
                state="readonly",
            ),
            "lzma2_block_size",
        ).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(fo, text="TIFF compression:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.tiff_compression = tk.StringVar(value=cfg.output.tiff_compression)
        _tip(
            ttk.Combobox(
                fo,
                textvariable=self.tiff_compression,
                values=["DEFLATE", "LZW", "LZMA", "ZSTD", "NONE"],
                width=8,
                state="readonly",
            ),
            "tiff_compression",
        ).grid(row=2, column=1, sticky=tk.W)

        ttk.Label(fo, text="CRS (EPSG):").grid(row=3, column=0, sticky=tk.W, pady=2)
        crs_frame = ttk.Frame(fo)
        crs_frame.grid(row=3, column=1, sticky=tk.W)
        no_crs_init = cfg.compression.crs.strip().lower() == "none"
        self.crs = tk.StringVar(value="" if no_crs_init else cfg.compression.crs)
        self._crs_entry = _tip(
            ttk.Combobox(
                crs_frame,
                textvariable=self.crs,
                width=14,
                values=["epsg:2180", "epsg:4326", "epsg:3857", "epsg:32633", "epsg:32634"],
            ),
            "crs_epsg",
        )
        self._crs_entry.pack(side=tk.LEFT)
        self.no_crs = tk.BooleanVar(value=no_crs_init)
        if no_crs_init:
            self._crs_entry.config(state="disabled")
        _tip(
            ttk.Checkbutton(
                crs_frame,
                text="No CRS (plain grid)",
                variable=self.no_crs,
                command=self._on_no_crs_change,
            ),
            "no_crs",
        ).pack(side=tk.LEFT, padx=6)

        self._add_spinbox(
            fo,
            4,
            "Decimal places:",
            "decimal_places",
            cfg.compression.decimal_places,
            0,
            6,
            "decimal_places",
        )

        self.delete_temp = tk.BooleanVar(value=cfg.output.delete_temp_files)
        _tip(
            ttk.Checkbutton(fo, text="Delete temp files", variable=self.delete_temp), "delete_temp"
        ).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=2)

        self.quiet = tk.BooleanVar(value=cfg.output.quiet)
        _tip(
            ttk.Checkbutton(fo, text="Quiet mode (suppress log output)", variable=self.quiet),
            "quiet",
        ).grid(row=6, column=0, columnspan=2, sticky=tk.W)

        self.show_log_panel = tk.BooleanVar(value=True)
        _tip(
            ttk.Checkbutton(
                fo,
                text="Show log panel",
                variable=self.show_log_panel,
                command=self._on_log_panel_toggle,
            ),
            "show_log_panel",
        ).grid(row=7, column=0, columnspan=2, sticky=tk.W)

    # =====================================================================
    # BUTTON BAR, PROGRESS, LOG
    # =====================================================================

    def _build_buttons(self, parent, row=1):
        f = ttk.Frame(parent)
        f.grid(row=row, column=0, pady=(6, 2), sticky=tk.W)
        self._run_button = _tip(
            ttk.Button(f, text="▶  Run", command=self._run, style="Run.TButton"), "run_button"
        )
        self._run_button.pack(side=tk.LEFT, padx=4)
        self._stop_button = _tip(
            ttk.Button(
                f, text="■  Stop", command=self._stop, style="Stop.TButton", state="disabled"
            ),
            "stop_button",
        )
        self._stop_button.pack(side=tk.LEFT, padx=4)
        _tip(ttk.Button(f, text="✕  Clear Log", command=self._clear_log), "clear_log").pack(
            side=tk.LEFT, padx=4
        )

    def _build_progress(self, parent, row=2):
        f = ttk.LabelFrame(parent, text="Progress", padding="6", style="Progress.TLabelframe")
        f.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 4))
        f.columnconfigure(0, weight=1)
        self._progress = _CanvasProgressBar(f, height=22)
        self._progress.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self._status_label = ttk.Label(f, text="Idle", foreground="gray", font=("TkDefaultFont", 8))
        self._status_label.grid(row=1, column=0, sticky=tk.W)

    def _build_log(self, parent, row=3):
        self._log_frame = ttk.LabelFrame(parent, text="Log", padding="8", style="Log.TLabelframe")
        self._log_frame.grid(row=row, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        self._log_frame.columnconfigure(0, weight=1)
        self._log_frame.rowconfigure(0, weight=1)
        self._log_text = tk.Text(
            self._log_frame,
            height=12,
            state="disabled",
            wrap="word",
            font=("TkFixedFont", 8),
            background="#1e1e1e",
            foreground="#d4d4d4",
            insertbackground="white",
        )
        self._log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        sb = ttk.Scrollbar(self._log_frame, command=self._log_text.yview)
        sb.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self._log_text.configure(yscrollcommand=sb.set)
        # Setup color tags for log output
        setup_log_text_colors(self._log_text)

    # =====================================================================
    # HELPER: labelled spinbox row
    # =====================================================================

    def _add_spinbox(self, parent, row, label, var_name, value, from_, to, tip_key=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        var = tk.DoubleVar(value=value) if isinstance(value, float) else tk.IntVar(value=int(value))
        setattr(self, var_name, var)
        spin = ttk.Spinbox(parent, from_=from_, to=to, textvariable=var, width=8)
        if tip_key:
            _tip(spin, tip_key)
        spin.grid(row=row, column=1, sticky=tk.W, pady=2)
        return spin

    def _on_uint8_mode_change(self, _event=None):
        """Enable only the 8-bit controls the selected method uses.

        dct: quality slider in Quality mode (JPEG for RGB); accuracy field and RGB channels in Accuracy mode.
        pred / sz3: the accuracy field only, because it sets the error bound of every band.
        """
        mode = self.uint8_mode.get()
        dct = self.method.get() == "dct"
        quality_on = dct and mode == "quality"
        accuracy_on = (not dct) or mode == "accuracy"
        rgb_on = dct and mode == "accuracy"

        self._rq_scale.config(state="normal" if quality_on else "disabled")
        self._rq_label_widget.config(state="normal" if quality_on else "disabled")
        self._u8acc_spin.config(state="normal" if accuracy_on else "disabled")
        self._u8acc_label_widget.config(state="normal" if accuracy_on else "disabled")
        self._u8rgb_label.config(state="normal" if rgb_on else "disabled")

    def _on_backend_change(self, event=None):  # pylint: disable=unused-argument
        """Handle backend change - enable/disable advanced heuristic."""
        backend = self.backend.get()
        if backend == "heuristic":
            self._adv_heuristic_cb.config(state="normal")
        else:
            self._adv_heuristic_cb.config(state="disabled")
            self.advanced_heuristic.set(False)


__all__ = ["GUIPanels"]
