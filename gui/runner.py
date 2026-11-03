"""
FIDWAC v2 GUI — compression / decompression runner
=====================================================
Mixin class GUIRunner providing:
  _run(), _stop(), _run_thread(), _run_finished()
  _process_file(), _process_directory(), _process_file_subprocess()
  _build_config()
  _validate()
  + log/progress helpers
  + file-browser callbacks
  + widget-state callbacks (_on_*)
"""

import os
import sys
import re as _re
import queue
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import tkinter as tk
from tkinter import messagebox, filedialog

from .widgets import _StreamToQueue, detect_log_tag


class GUIRunner:
    """Mixin class for compression/decompression runner."""

    def __init__(self):
        """Initialize GUIRunner attributes to None."""
        self.auto_sample_size = None
        self.std_high = None
        self.std_medium = None
        self.decimal_places = None
        self._output_label = None
        self._op_label = None
        self.input_path = None
        self.accuracy = None
        self.method = None
        self.block_size = None
        self.backend = None
        self.uint8_mode = None
        self.rgb_quality = None
        self.processes = None
        self._log_queue = None
        self.root = None
        self._rq_label = None
        self._log_text = None
        self.output_dir = None
        self.auto_select = None
        self._block_size_cb = None
        self._rq_scale = None
        self._rq_label_widget = None
        self._u8acc_spin = None
        self._u8acc_label_widget = None
        self._u8acc_hint = None
        self._u8rgb_label = None
        self._backend_cb = None
        self.allow_n8 = None
        self.allow_n16 = None
        self.allow_n32 = None
        self.show_log_panel = None
        self._log_frame = None
        self.no_crs = None
        self._crs_entry = None
        self.input_type = None
        self._cpu_count = None
        self.uint8_accuracy = None
        self.rgb_ch_r = None
        self.rgb_ch_g = None
        self.rgb_ch_b = None
        self.quiet = None
        self.source_dir = None
        self.results_dir = None
        self.advanced_heuristic = None
        self.accept_prediction = None
        self.minimize_backscan = None
        self.backscan_break = None
        self.ycbcr_y_mult = None
        self.ycbcr_cb_mult = None
        self.ycbcr_cr_mult = None
        self.compression_method = None
        self.tiff_compression = None
        self.delete_temp = None
        self.crs = None
        self._stop_event = None
        self._stop_button = None
        self._status_label = None
        self._run_button = None
        self._progress = None
        self.files_parallel = None
        self._minimize_backscan_spin = None
        self._backscan_break_spin = None

    # =====================================================================
    # LOG HELPERS
    # =====================================================================

    def _log(self, message):
        self._log_queue.put(str(message))

    def _poll_log(self):
        try:
            while True:
                msg = self._log_queue.get_nowait()
                line = msg[1] if isinstance(msg, tuple) else str(msg)
                if line:
                    try:
                        sys.__stdout__.write(line + "\n")
                        sys.__stdout__.flush()
                    except Exception:
                        pass
                    self._log_text.config(state="normal")
                    # Detect tag and apply color
                    tag = detect_log_tag(line)
                    self._log_text.insert(tk.END, line + "\n", tag)
                    self._log_text.see(tk.END)
                    self._log_text.config(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log)

    def _clear_log(self):
        self._log_text.config(state="normal")
        self._log_text.delete(1.0, tk.END)
        self._log_text.config(state="disabled")

    # =====================================================================
    # CALLBACKS
    # =====================================================================

    def _on_input_changed(self, *_):
        path = self.input_path.get()

        # Check if file exists and determine operation
        if path.lower().endswith((".7z", ".fpred", ".sz3")):
            if os.path.exists(path):
                self._op_label.config(text="✓ Decompression", foreground="#4ec94e")
            else:
                self._op_label.config(text="❌ File not found", foreground="#ff6b6b")
        elif path:
            if os.path.exists(path):
                if os.path.isfile(path):
                    self._op_label.config(text="✓ Compression (file)", foreground="#89c3e8")
                else:
                    self._op_label.config(text="✓ Compression (directory)", foreground="#89c3e8")
            else:
                self._op_label.config(text="❌ Path not found", foreground="#ff6b6b")
        else:
            self._op_label.config(text="")

    def _on_output_changed(self, *_):
        path = self.output_dir.get()
        if path:
            if os.path.exists(path):
                self._output_label.config(text="✓ Valid output directory", foreground="#4ec94e")
            else:
                self._output_label.config(text="❌ Directory doesn't exist", foreground="#ff6b6b")
        else:
            self._output_label.config(text="")

    def _on_auto_select_change(self):
        state = "disabled" if self.auto_select.get() else "readonly"
        self._block_size_cb.config(state=state)

    def _on_uint8_mode_change(self):
        mode = self.uint8_mode.get()
        _gray = "#999999"
        if mode == "quality":
            self._rq_scale.state(["!disabled"])
            self._rq_label_widget.config(text="Quality (%):", foreground="")
            self._rq_label.config(text=f"{self.rgb_quality.get()}%", foreground="")
            self._u8acc_spin.config(state="disabled")
            self._u8acc_label_widget.config(foreground=_gray)
            self._u8acc_hint.config(foreground=_gray)
            self._u8rgb_label.config(foreground=_gray)
            self._backend_cb.config(state="readonly")
            # Re-enable backscan controls (if created)
            if hasattr(self, "minimize_backscan"):
                self._minimize_backscan_spin.config(state="normal")
                self._backscan_break_spin.config(state="normal")
        elif mode == "accuracy":
            self._rq_scale.state(["disabled"])
            self._rq_label_widget.config(text="Quality (N/A):", foreground=_gray)
            self._rq_label.config(text="—", foreground=_gray)
            self._u8acc_spin.config(state="normal")
            self._u8acc_label_widget.config(foreground="")
            self._u8acc_hint.config(foreground="gray")
            self._u8rgb_label.config(foreground="")
            # Allow heuristic backend for uint8 accuracy mode (needed for cm=6 RGB YCbCr)
            self._backend_cb.config(state="readonly")
            self._on_backend_change()
            # Keep backscan controls enabled (if created)
            if hasattr(self, "minimize_backscan"):
                self._minimize_backscan_spin.config(state="normal")
                self._backscan_break_spin.config(state="normal")
        else:  # lossless
            self._rq_scale.state(["disabled"])
            self._rq_label_widget.config(text="Quality (N/A):", foreground=_gray)
            self._rq_label.config(text="—", foreground=_gray)
            self._u8acc_spin.config(state="disabled")
            self._u8acc_label_widget.config(foreground=_gray)
            self._u8acc_hint.config(foreground=_gray)
            self._u8rgb_label.config(foreground=_gray)
            self._backend_cb.config(state="disabled")
            # Lossless uses PNG, not DCT - disable backscan controls (if created)
            if hasattr(self, "minimize_backscan"):
                self._minimize_backscan_spin.config(state="disabled")
                self._backscan_break_spin.config(state="disabled")

    def _on_allowed_bs_change(self):
        allowed = []
        if self.allow_n8.get():
            allowed.append(8)
        if self.allow_n16.get():
            allowed.append(16)
        if self.allow_n32.get():
            allowed.append(32)

    def _on_log_panel_toggle(self):
        if self.show_log_panel.get():
            self._log_frame.grid()
        else:
            self._log_frame.grid_remove()
        self.root.update_idletasks()

    def _on_no_crs_change(self):
        state = "disabled" if self.no_crs.get() else "normal"
        self._crs_entry.config(state=state)

    def _on_backend_change(self, _event=None):
        backend = self.backend.get()
        if hasattr(self, "advanced_heuristic"):
            if backend == "binary":
                self.advanced_heuristic.set(False)
                self.accept_prediction.set(False)
            elif backend == "heuristic":
                self.advanced_heuristic.set(True)

        if hasattr(self, "_adv_heuristic_cb"):
            h_state = "disabled" if backend == "binary" else "normal"
            for w in [self._adv_heuristic_cb, self._accept_pred_cb]:
                w.config(state=h_state)
            for w in [self._minimize_backscan_spin, self._backscan_break_spin]:
                w.config(state="normal")

    # =====================================================================
    # FILE BROWSERS
    # =====================================================================

    def _browse_input(self):
        if self.input_type.get() == "file":
            path = filedialog.askopenfilename(
                title="Select input file",
                filetypes=[
                    ("Supported rasters", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.asc *.img *.7z *.fpred *.sz3"),
                    ("GeoTIFF", "*.tif *.tiff"),
                    ("PNG + world file", "*.png"),
                    ("JPEG + world file", "*.jpg *.jpeg"),
                    ("BMP + world file", "*.bmp"),
                    ("Erdas Imagine", "*.img"),
                    ("ASC Grid", "*.asc"),
                    ("Archive / Compressed", "*.7z *.fpred *.sz3"),
                    ("All files", "*.*"),
                ],
            )
        else:
            path = filedialog.askdirectory(title="Select input directory")
        if path:
            self.input_path.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_dir.set(path)

    # =====================================================================
    # VALIDATION
    # =====================================================================

    def _validate(self):
        result = True
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input path.")
            result = False
        elif not os.path.exists(self.input_path.get()):
            messagebox.showerror("Error", f"Input path does not exist:\n{self.input_path.get()}")
            result = False
        elif not self.output_dir.get():
            messagebox.showerror("Error", "Please select an output directory.")
            result = False
        elif not os.path.exists(self.output_dir.get()):
            messagebox.showerror(
                "Error", f"Output directory does not exist:\n{self.output_dir.get()}"
            )
            result = False
        else:
            acc = self.accuracy.get()
            if not 0.001 <= acc <= 1.0:
                messagebox.showerror("Error", "Accuracy must be between 0.001 and 1.0.")
                result = False
            else:
                proc = self.processes.get()
                if not 1 <= proc <= self._cpu_count:
                    messagebox.showerror("Error", f"Processes must be between 1 and {self._cpu_count}.")
                    result = False
        return result

    # =====================================================================
    # BUILD CONFIG FROM GUI
    # =====================================================================

    def _build_config(self):
        from config import load_config

        config = load_config(None)

        from config import set_gpu_mode

        config.performance.gpu = self.gpu.get()
        set_gpu_mode(config.performance.gpu)  # the in-process batch path reads this mode
        config.compression.accuracy = self.accuracy.get()
        config.compression.auto_select_block_size = self.auto_select.get()
        config.compression.rgb_quality = self.rgb_quality.get()
        _mode = self.uint8_mode.get()
        config.compression.lossless = _mode == "lossless"
        config.compression.uint8_accuracy_mode = _mode == "accuracy"
        config.compression.uint8_accuracy = self.uint8_accuracy.get()
        _r, _g, _b = self.rgb_ch_r.get(), self.rgb_ch_g.get(), self.rgb_ch_b.get()
        config.compression.rgb_channel_indices = (
            [_r, _g, _b] if (_r > 0 and _g > 0 and _b > 0) else []
        )
        allowed = []
        if self.allow_n8.get():
            allowed.append(8)
        if self.allow_n16.get():
            allowed.append(16)
        if self.allow_n32.get():
            allowed.append(32)
        config.compression.allowed_block_sizes = allowed
        if not self.auto_select.get():
            config.compression.block_size = int(self.block_size.get())
        config.model.backend = self.backend.get()

        config.compression.auto_select_sample_size = int(self.auto_sample_size.get())
        config.compression.auto_select_std_threshold_high = float(self.std_high.get())
        config.compression.auto_select_std_threshold_medium = float(self.std_medium.get())
        config.output.quiet = self.quiet.get()
        config.compression.method = self.method.get()
        config.directories.source = self.source_dir.get()
        config.directories.results = self.results_dir.get()
        config.model.advanced_heuristic = self.advanced_heuristic.get()
        config.model.accept_prediction_if_within_accuracy = self.accept_prediction.get()
        config.model.minimize_backscan = self.minimize_backscan.get()
        config.model.backscan_break_after = self.backscan_break.get()
        config.compression.ycbcr_y_multiplier = self.ycbcr_y_mult.get()
        config.compression.ycbcr_cb_multiplier = self.ycbcr_cb_mult.get()
        config.compression.ycbcr_cr_multiplier = self.ycbcr_cr_mult.get()
        config.output.compression_method = self.compression_method.get()
        config.output.lzma2_block_size = self.lzma2_block_size.get().strip()
        config.output.tiff_compression = self.tiff_compression.get()
        config.output.delete_temp_files = self.delete_temp.get()
        config.compression.crs = "none" if self.no_crs.get() else self.crs.get()
        config.compression.decimal_places = self.decimal_places.get()

        return config

    # =====================================================================
    # RUN / STOP
    # =====================================================================

    def _stop(self):
        import signal
        import fnmatch

        self._stop_event.set()
        self._log("Stop requested — killing child processes…")
        self._stop_button.config(state="disabled")

        my_pid = os.getpid()
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat", encoding="utf-8") as f:
                    ppid = int(f.read().split()[3])
                if ppid == my_pid:
                    os.kill(int(entry), signal.SIGTERM)
            except (OSError, ValueError, ProcessLookupError):
                pass

        for cmd in ["7zz", "7zip", "7z"]:
            try:
                subprocess.run(["pkill", "-9", "-f", cmd], capture_output=True, check=False)
            except Exception:  # pylint: disable=broad-except
                pass
        try:
            subprocess.run(["pkill", "-9", "-f", "compress.py"], capture_output=True, check=False)
        except Exception:
            pass

        output_dir = self.output_dir.get()
        if output_dir and os.path.isdir(output_dir):
            for name in os.listdir(output_dir):
                if any(
                    fnmatch.fnmatch(name, p)
                    for p in [
                        "_fidwac_ch_*.tmp",
                        "_fidwac_rgb_*.bin",
                        "_fidwac_meta_*.msgpack",
                        "*.msgpack",
                    ]
                ):
                    try:
                        os.remove(os.path.join(output_dir, name))
                        self._log(f"  Deleted temp: {name}")
                    except OSError:
                        pass

        self._status_label.config(text="Stopped", foreground="#8d6e63")
        self._run_button.config(state="normal")
        self._stop_button.config(state="disabled")

    def _run(self):
        self._commit_accuracy()  # pylint: disable=no-member
        if not self._validate():
            return
        self._stop_event.clear()
        self._reset_stats()  # pylint: disable=no-member
        self._run_button.config(state="disabled")
        self._stop_button.config(state="normal")
        self._progress.reset()
        self._status_label.config(text="Starting…", foreground="#555555")
        threading.Thread(target=self._run_thread, daemon=True).start()

    def _update_progress(self, pct, status_text=""):
        def _do(v=pct, t=status_text):
            self._progress.set(v)
            if t:
                lm = _re.match(r"^(.*?)\s*\d+%", t)
                label = lm.group(1).rstrip(": ").strip() if lm else ""
                cm = _re.search(r"(\d+)/(\d+)", t)
                count = f"  {cm.group(1)}/{cm.group(2)}" if cm else ""
                self._status_label.config(
                    text=f"{label}{count}" if label else t[:70],
                    foreground="#1a1a1a",
                )

        self.root.after(0, _do)

    def _run_thread(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = _StreamToQueue(self._log_queue, self._update_progress, old_stdout)
        sys.stderr = _StreamToQueue(self._log_queue, self._update_progress, old_stderr)
        os.environ.setdefault("TQDM_ASCII", "false")
        os.environ.setdefault("TQDM_NCOLS", "80")
        try:
            config = self._build_config()
            num_processes = self.processes.get()
            input_path = self.input_path.get()
            output_dir = self.output_dir.get()

            self._log("=" * 60)
            if os.path.isfile(input_path):
                self._process_file(input_path, output_dir, config, num_processes)
            else:
                self._process_directory(input_path, output_dir, config, num_processes)
            self._log("=" * 60)
            self._log("Stopped." if self._stop_event.is_set() else "Done!")

            # Save config to persist GUI settings for next run
            if not self._stop_event.is_set():
                from config import save_config

                save_config(config, "config.json")
                self._log("Settings saved to config.json")
        except Exception as exc:
            self._log(f"Error: {exc}")
            self.root.after(0, lambda e=exc: messagebox.showerror("Error", str(e)))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            os.environ.pop("TQDM_ASCII", None)
            os.environ.pop("TQDM_NCOLS", None)
            self.root.after(0, self._run_finished)

    def _run_finished(self):
        if getattr(self._progress, "_pct", 0) >= 100:
            self._status_label.config(text="Done", foreground="#2e7d32")
        elif self._stop_event.is_set():
            self._status_label.config(text="Stopped", foreground="#8d6e63")
        elif getattr(self._progress, "_pct", 0) > 0:
            self._status_label.config(text="Finished", foreground="#1a1a1a")
        else:
            self._status_label.config(text="Idle", foreground="gray")
        self._run_button.config(state="normal")
        self._stop_button.config(state="disabled")
        self._stop_event.clear()

    # =====================================================================
    # FILE PROCESSING
    # =====================================================================

    def _process_file(self, input_path, output_dir, config, num_processes):
        from compress.decompression import decompress_file
        from compress.compression import compress_image

        if input_path.lower().endswith((".7z", ".fpred", ".sz3")):
            self._log(f"Decompressing: {input_path}")
            out = decompress_file(input_path, config, output_dir)
        elif config.compression.method in ("pred", "sz3"):
            self._log(f"Compressing with {config.compression.method}: {input_path}")
            compress_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "compress.py")
            _, out = self._process_file_subprocess(
                input_path, output_dir, compress_py, config, num_processes
            )
            self._update_stats_file(input_path, out)  # pylint: disable=no-member
        else:
            self._log(f"Compressing: {input_path}")
            out = compress_image(input_path, config, num_processes, output_dir)
            self._update_stats_file(input_path, out)  # pylint: disable=no-member
        self._log(f"Output: {out}")

    def _process_directory(self, input_dir, output_dir, config, num_processes):
        from compress.decompression import decompress_file
        from compress.compression import compress_image, find_raster_files

        input_dir = os.path.normpath(input_dir)
        raster_files = find_raster_files(input_dir, recursive=True)
        archive_files = []
        for root, _dirs, fnames in os.walk(input_dir):
            for fname in fnames:
                if fname.lower().endswith((".7z", ".fpred", ".sz3")):
                    archive_files.append(os.path.join(root, fname))

        files = sorted(set(raster_files + archive_files))
        if not files:
            self._log("No files found in directory.")
            return

        total_files = len(files)
        files_parallel = self.files_parallel.get()

        def _file_output_dir(path):
            rel = os.path.relpath(os.path.dirname(path), input_dir)
            d = output_dir if rel == "." else os.path.join(output_dir, rel)
            os.makedirs(d, exist_ok=True)
            return d

        if files_parallel <= 1:
            self._log(f"Found {total_files} files — 1 file at a time, {num_processes} processes")
            self._update_progress(0, f"0/{total_files} files 0%")
            for i, path in enumerate(files, 1):
                if self._stop_event.is_set():
                    self._log(f"  Stopped after {i - 1}/{total_files} files.")
                    break
                self._log(f"[{i}/{total_files}] {os.path.relpath(path, input_dir)}")
                pct_before = int((i - 1) / total_files * 100)
                self._update_progress(
                    pct_before,
                    f"{i-1}/{total_files} files {pct_before}%|" f"{'█'*int(pct_before/5)}|",
                )
                try:
                    file_out_dir = _file_output_dir(path)
                    if path.lower().endswith((".7z", ".fpred", ".sz3")):
                        out = decompress_file(path, config, file_out_dir)
                    elif config.compression.method in ("pred", "sz3"):
                        compress_py = os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), "..", "compress.py"
                        )
                        _, out = self._process_file_subprocess(
                            path, file_out_dir, compress_py, config, num_processes
                        )
                        self._update_stats_file(path, out)  # pylint: disable=no-member
                    else:
                        out = compress_image(path, config, num_processes, file_out_dir)
                        self._update_stats_file(path, out)  # pylint: disable=no-member
                    self._log(f"  → {os.path.relpath(out, output_dir)}")
                except Exception as exc:
                    self._log(f"  Error: {exc}")
                    self._record_error()  # pylint: disable=no-member
                pct = int(i / total_files * 100)
                self._update_progress(pct, f"{i}/{total_files} files {pct}%|{'█'*int(pct/5)}|")
            return

        processes_per_file = max(1, num_processes // files_parallel)
        max_workers = min(files_parallel, total_files)
        self._log(
            f"Found {total_files} files — {files_parallel} parallel, "
            f"{processes_per_file} proc each"
        )
        self._update_progress(0, f"0/{total_files} files 0%||")

        compress_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "compress.py")
        completed_count = 0
        tasks = [(p, _file_output_dir(p)) for p in files]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx, (path, file_output_dir) in enumerate(tasks, 1):
                if self._stop_event.is_set():
                    break
                self._log(f"[{idx}/{total_files}] {os.path.relpath(path, input_dir)}")
                future = executor.submit(
                    self._process_file_subprocess,
                    path,
                    file_output_dir,
                    compress_py,
                    config,
                    processes_per_file,
                )
                futures[future] = path

            for future in as_completed(futures):
                if self._stop_event.is_set():
                    break
                completed_count += 1
                pct = int(completed_count / total_files * 100)
                self._update_progress(
                    pct,
                    f"{completed_count}/{total_files} files {pct}%|" f"{'█'*int(pct/5)}|",
                )
                try:
                    result_path, result_out = future.result()
                    self._update_stats_file(result_path, result_out)  # pylint: disable=no-member
                    self._log(f"  → {os.path.relpath(result_out, output_dir)}")
                except Exception as exc:
                    self._log(f"  Error ({os.path.basename(futures[future])}): {exc}")
                    self._record_error()  # pylint: disable=no-member

    @staticmethod
    def _is_uint8_raster(path):
        """True when the file is a raster and every band is uint8."""
        try:
            import rasterio  # pylint: disable=import-outside-toplevel

            with rasterio.open(path) as src:
                return all(dt == "uint8" for dt in src.dtypes)
        except Exception:  # pylint: disable=broad-except
            return False

    def _process_file_subprocess(self, input_path, output_dir, compress_py, config, num_processes):
        import subprocess as sp
        from compress.decompression import decompress_file

        if input_path.lower().endswith((".7z", ".fpred", ".sz3")):
            out = decompress_file(input_path, config, output_dir)
            return input_path, out

        cmd = [sys.executable, compress_py, "-i", input_path, "-o", output_dir]
        cmd += ["--method", config.compression.method]
        accuracy = config.compression.accuracy
        if config.compression.method in ("pred", "sz3") and self._is_uint8_raster(input_path):
            # 8-bit input: the Accuracy field of the 8-bit section (pixel values), not the float slider
            accuracy = config.compression.uint8_accuracy
        cmd += ["--accuracy", str(accuracy)]
        if config.compression.auto_select_block_size:
            cmd += ["--auto"]
        else:
            cmd += ["--block-size", str(config.compression.block_size)]
        cmd += ["--backend", config.model.backend]
        cmd += ["--minimize-backscan", str(config.model.minimize_backscan)]
        cmd += ["--backscan-break-after", str(config.model.backscan_break_after)]
        if config.model.advanced_heuristic:
            cmd += ["--advanced-heuristic"]
        else:
            cmd += ["--no-advanced-heuristic"]
        if config.model.accept_prediction_if_within_accuracy:
            cmd += ["--accept-prediction"]
        else:
            cmd += ["--no-accept-prediction"]
        cmd += ["--decimal-places", str(config.compression.decimal_places)]
        cmd += ["--processes", str(num_processes), "--gpu", self.gpu.get(), "--quiet"]
        if config.compression.uint8_accuracy_mode:
            cmd += [
                "--uint8-accuracy-mode",
                "--uint8-accuracy",
                str(config.compression.uint8_accuracy),
            ]
        if config.compression.rgb_quality != 85:
            cmd += ["--rgb-quality", str(config.compression.rgb_quality)]
        if config.compression.lossless:
            cmd += ["--lossless"]
        _rgb_idx = getattr(config.compression, "rgb_channel_indices", [])
        if _rgb_idx and len(_rgb_idx) == 3:
            cmd += ["--rgb-channel-indices", ",".join(str(x) for x in _rgb_idx)]
            if getattr(config.compression, "ycbcr_per_block_mode", False):
                cmd += ["--ycbcr-per-block"]
        if config.compression.uint8_scaling_factor != 1:
            cmd += ["--scaling-factor", str(config.compression.uint8_scaling_factor)]
        cmd += ["--ycbcr-y-multiplier", str(config.compression.ycbcr_y_multiplier)]
        cmd += ["--ycbcr-cb-multiplier", str(config.compression.ycbcr_cb_multiplier)]
        cmd += ["--ycbcr-cr-multiplier", str(config.compression.ycbcr_cr_multiplier)]
        if config.compression.allowed_block_sizes:
            cmd += [
                "--allowed-block-sizes",
                ",".join(str(n) for n in config.compression.allowed_block_sizes),
            ]
        cmd += ["--crs", config.compression.crs]
        cmd += ["--compression-method", config.output.compression_method]
        if config.output.lzma2_block_size:
            cmd += ["--lzma2-block-size", config.output.lzma2_block_size]
        cmd += [
            (
                "--verify-with-full-idct"
                if config.performance.verify_with_full_idct
                else "--no-verify-with-full-idct"
            )
        ]
        cmd += ["--l2-precheck" if config.performance.l2_precheck_enabled else "--no-l2-precheck"]
        cmd += [
            (
                "--incremental-backscan"
                if config.performance.incremental_backscan
                else "--no-incremental-backscan"
            )
        ]

        result = sp.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
        if result.returncode != 0:
            # Handled CLI errors are printed on stdout ("Error ..."); tracebacks go to stderr.
            errors = [ln.strip() for ln in result.stdout.splitlines() if ln.startswith("Error")]
            tail = [ln.strip() for ln in result.stderr.splitlines() if ln.strip()]
            raise RuntimeError(
                errors[-1] if errors else (tail[-1] if tail else f"exit {result.returncode}")
            )

        # Return the file the CLI wrote for this method (.7z, .fpred or .sz3), newest first.
        stem = os.path.splitext(os.path.basename(input_path))[0]
        exts = {"pred": (".fpred", ".7z"), "sz3": (".sz3", ".7z")}.get(
            config.compression.method, (".7z",)
        )
        found = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.startswith(stem) and f.endswith(exts)
        ]
        if not found:
            raise RuntimeError(f"compression finished but no output found in {output_dir}")
        return input_path, max(found, key=os.path.getmtime)


__all__ = ["GUIRunner"]
