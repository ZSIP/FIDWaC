#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIDWAC v2 - Predictive vs Binary Compression Benchmark Tool
===========================================================
This tool runs batch compression benchmarks on a directory of TIFF images,
comparing the 'binary' (exact) search with 'heuristic' (advanced) search modes.
Produces detailed CSV results, summaries, and comparison charts.
"""

import os
import re
import sys
import time
import csv
import threading
import subprocess
import concurrent.futures
from typing import List, Dict, Any, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Import matplotlib for plot generation
try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Resolve absolute path to compress.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPRESS_SCRIPT = os.path.join(PROJECT_ROOT, "compress.py")


class BenchmarkGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("FIDWAC v2 - Compression Benchmark Tool")
        self.root.geometry("1100x750")
        self.root.minimum_size = (900, 600)
        
        # Benchmarking state
        self.input_dir = tk.StringVar()
        self.accuracy_float = tk.StringVar(value="0.05")
        self.accuracy_uint8 = tk.StringVar(value="5")
        self.block_size = tk.StringVar(value="8")
        self.num_processes = tk.StringVar(value="auto")
        self.rgb_channel_indices = tk.StringVar(value="")
        self.delete_archives = tk.BooleanVar(value=True)
        self.is_running = False
        self.results_detailed: List[Dict[str, Any]] = []
        self.results_summary: List[Dict[str, Any]] = []
        self.results_lock = threading.Lock()
        
        self.setup_ui()
        
    def setup_ui(self):
        # Apply style
        style = ttk.Style()
        style.theme_use("clam")
        
        # Main Frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            header_frame, 
            text="FIDWAC v2 Benchmark: Binary vs Advanced Heuristic", 
            font=("Arial", 16, "bold")
        ).pack(anchor=tk.W)
        ttk.Label(
            header_frame, 
            text="Run batch comparison of compression ratio, time, and accuracy across multiple TIFF rasters.",
            font=("Arial", 10, "italic")
        ).pack(anchor=tk.W)
        
        # Config Frame (Left Side) & Log Frame (Right Side)
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = ttk.Frame(content_frame, width=350)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)
        
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Left Side: File/Folder Selection
        files_lf = ttk.LabelFrame(left_frame, text="1. Select Input", padding="10")
        files_lf.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(files_lf, text="Directory of TIFF files:").pack(anchor=tk.W)
        dir_sel_frame = ttk.Frame(files_lf)
        dir_sel_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(dir_sel_frame, textvariable=self.input_dir, width=25).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(dir_sel_frame, text="Browse...", command=self.browse_dir).pack(side=tk.LEFT)
        
        # Left Side: Parameters
        params_lf = ttk.LabelFrame(left_frame, text="2. Compression Parameters", padding="10")
        params_lf.pack(fill=tk.X, pady=(0, 10))
        
        # Accuracy Float32
        ttk.Label(params_lf, text="Float32 Accuracy (e.g., 0.05):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(params_lf, textvariable=self.accuracy_float, width=10).grid(row=0, column=1, sticky=tk.E, pady=5)
        
        # Accuracy Uint8
        ttk.Label(params_lf, text="Uint8 Accuracy (2 to 20):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(params_lf, textvariable=self.accuracy_uint8, width=10).grid(row=1, column=1, sticky=tk.E, pady=5)
        
        # Block Size
        ttk.Label(params_lf, text="Block size:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(
            params_lf, 
            textvariable=self.block_size, 
            values=["8", "16", "32", "auto"], 
            width=8, 
            state="readonly"
        ).grid(row=2, column=1, sticky=tk.E, pady=5)
        
        # Num Processes
        ttk.Label(params_lf, text="CPU Workers:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(params_lf, textvariable=self.num_processes, width=10).grid(row=3, column=1, sticky=tk.E, pady=5)

        # RGB Channel Indices (for cm=6 per-block YCbCr)
        ttk.Label(params_lf, text="RGB Channels (e.g. 1,2,3):").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Entry(params_lf, textvariable=self.rgb_channel_indices, width=10).grid(row=4, column=1, sticky=tk.E, pady=5)

        # Delete archives option
        ttk.Checkbutton(
            params_lf,
            text="Delete temp .7z files after test",
            variable=self.delete_archives
        ).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=10)
        
        # Left Side: Run Action
        self.btn_run = ttk.Button(left_frame, text="START BENCHMARK", style="Accent.TButton", command=self.start_benchmark)
        self.btn_run.pack(fill=tk.X, pady=10, ipady=5)
        
        self.progress_bar = ttk.Progressbar(left_frame, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.status_lbl = ttk.Label(left_frame, text="Status: Ready", font=("Arial", 10, "bold"))
        self.status_lbl.pack(anchor=tk.W, pady=5)
        
        # Right Side: Tabbed Interface (Console / Charts / Detailed)
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Log Console
        log_tab = ttk.Frame(self.notebook)
        self.notebook.add(log_tab, text="Console Log")
        
        self.log_txt = tk.Text(log_tab, wrap=tk.WORD, font=("Consolas", 9), bg="#1e1e1e", fg="#ffffff")
        self.log_txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(log_tab, orient=tk.VERTICAL, command=self.log_txt.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_txt.config(yscrollcommand=scrollbar.set)
        
        # Tab 2: Detailed Results Table
        results_tab = ttk.Frame(self.notebook)
        self.notebook.add(results_tab, text="Summary Table")
        
        columns = ("filename", "orig_mb", "bin_ratio", "heur_no_accept_ratio", "heur_accept_ratio", "bin_time", "heur_no_accept_time", "heur_accept_time")
        self.tree = ttk.Treeview(results_tab, columns=columns, show="headings")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.tree.heading("filename", text="File Name")
        self.tree.heading("orig_mb", text="Orig Size (MB)")
        self.tree.heading("bin_ratio", text="Binary Ratio")
        self.tree.heading("heur_no_accept_ratio", text="Heur No Accept Ratio")
        self.tree.heading("heur_accept_ratio", text="Heur Accept Ratio")
        self.tree.heading("bin_time", text="Binary Time (s)")
        self.tree.heading("heur_no_accept_time", text="Heur No Accept (s)")
        self.tree.heading("heur_accept_time", text="Heur Accept (s)")
        
        # Tree column widths
        self.tree.column("filename", width=180, anchor=tk.W)
        for col in columns[1:]:
            self.tree.column(col, width=90, anchor=tk.CENTER)
            
        tree_scroll = ttk.Scrollbar(results_tab, orient=tk.VERTICAL, command=self.tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.config(yscrollcommand=tree_scroll.set)
        
        # Tab 3: Charts Canvas
        self.charts_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.charts_tab, text="Comparison Charts")
        self.charts_canvas = None
        
        if not HAS_MATPLOTLIB:
            lbl_err = ttk.Label(
                self.charts_tab, 
                text="Matplotlib is not installed.\nCharts cannot be rendered directly in GUI,\nbut CSV files will be successfully generated.",
                justify=tk.CENTER,
                font=("Arial", 12)
            )
            lbl_err.pack(expand=True)
            
        # Log welcome message
        self.log("=== FIDWAC v2 Compression Benchmarking Tool ===")
        self.log("Ready to execute. Select a TIFF folder to begin.")
        if not HAS_MATPLOTLIB:
            self.log("Note: Matplotlib not found. Chart generation disabled inside GUI.")
            
    def log(self, msg: str):
        self.root.after(0, lambda: self._log_main_thread(msg))

    def _log_main_thread(self, msg: str):
        self.log_txt.insert(tk.END, msg + "\n")
        self.log_txt.see(tk.END)
        
    def browse_dir(self):
        directory = filedialog.askdirectory(title="Select Folder with GeoTIFF files")
        if directory:
            self.input_dir.set(directory)
            self.log(f"Selected input directory: {directory}")
            
    def start_benchmark(self):
        if self.is_running:
            return
            
        directory = self.input_dir.get()
        if not directory or not os.path.isdir(directory):
            messagebox.showerror("Error", "Please select a valid input directory containing TIFF files.")
            return
            
        # Validate parameters
        try:
            float(self.accuracy_float.get())
            int(self.accuracy_uint8.get())
        except ValueError:
            messagebox.showerror("Error", "Accuracy values must be valid numeric values.")
            return
            
        self.is_running = True
        self.btn_run.config(state="disabled")
        self.status_lbl.config(text="Status: Executing benchmarks...")
        self.progress_bar["value"] = 0
        
        # Clear logs and tables
        self.log_txt.delete("1.0", tk.END)
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Run benchmark in background thread
        thread = threading.Thread(target=self.run_benchmark_worker, daemon=True)
        thread.start()
        
    def run_benchmark_worker(self):
        directory = self.input_dir.get()
        acc_float = self.accuracy_float.get()
        acc_uint8 = self.accuracy_uint8.get()
        bs = self.block_size.get()
        workers = self.num_processes.get()
        rgb_ch = self.rgb_channel_indices.get()
        delete_temp = self.delete_archives.get()
        
        self.log("=== STARTING BATCH BENCHMARK ===")
        self.log(f"Input directory: {directory}")
        self.log(f"Float32 Accuracy: {acc_float} | Uint8 Accuracy: {acc_uint8}")
        self.log(f"Block size: {bs} | Processes: {workers}")
        self.log(f"Clean temp archives: {'YES' if delete_temp else 'NO'}\n")
        
        # Find TIFF files
        tiff_files = []
        for f in os.listdir(directory):
            if f.lower().endswith((".tif", ".tiff")):
                tiff_files.append(os.path.join(directory, f))
                
        if not tiff_files:
            self.log("❌ ERROR: No TIFF files found in the selected folder.")
            self.root.after(0, self.benchmark_finished, "No TIFF files found.")
            return
            
        self.log(f"Found {len(tiff_files)} TIFF file(s) to process.\n")
        
        # Initialize results lists
        self.results_detailed = []
        self.results_summary = []
        
        total_steps = len(tiff_files) * 3  # 3 modes per file
        completed_steps = 0
        
        # Core dividing logic
        try:
            total_cores = os.cpu_count() or 4
        except Exception:
            total_cores = 4

        gui_workers = workers.strip()
        if gui_workers.isdigit():
            cores_to_use = int(gui_workers)
        else:
            cores_to_use = total_cores

        sub_workers = max(1, cores_to_use // 5)
        sub_workers_str = str(sub_workers)
        
        self.log(f"ℹ️ Dividing CPU cores/workers: overall using up to 5 concurrent file workers, each file with --processes {sub_workers_str} (out of {cores_to_use} available/requested cores).\n")

        def process_single_file(filepath: str, idx: int):
            nonlocal completed_steps
            filename = os.path.basename(filepath)
            orig_size = os.path.getsize(filepath)
            orig_mb = orig_size / (1024 * 1024)
            
            self.log(f"--- [File {idx+1}/{len(tiff_files)}] Started: {filename} ({orig_mb:.2f} MB) ---")
            
            # Temporary output path for 7z archives
            bin_7z_path = os.path.join(directory, f"temp_bench_binary_{filename}.7z")
            heur_no_accept_7z_path = os.path.join(directory, f"temp_bench_heur_no_accept_{filename}.7z")
            heur_accept_7z_path = os.path.join(directory, f"temp_bench_heur_accept_{filename}.7z")

            # Mode 1: Binary Search
            self.log(f"  [{filename}] Running mode: BINARY search (using {sub_workers_str} processes)...")
            bin_time, bin_size, bin_err, bin_valid, bin_status = self.execute_compression(
                filepath, bin_7z_path, "binary", False, False, acc_float, acc_uint8, bs, sub_workers_str, rgb_ch
            )
            with self.results_lock:
                completed_steps += 1
                self.update_progress(completed_steps, total_steps)

            if bin_status == "Success":
                bin_ratio = orig_size / bin_size if bin_size > 0 else 0
                bin_speed = orig_mb / bin_time if bin_time > 0 else 0
                self.log(f"  [{filename}] ✅ Binary ratio: {bin_ratio:.2f}x | Speed: {bin_speed:.2f} MB/s | MaxErr: {bin_err} | Valid: {bin_valid}")
            else:
                self.log(f"  [{filename}] ❌ Binary FAILED: {bin_status}")
                bin_ratio, bin_speed = 0, 0

            # Mode 2: Advanced Heuristic WITHOUT accept
            self.log(f"  [{filename}] Running mode: HEURISTIC (Advanced) WITHOUT accept (using {sub_workers_str} processes)...")
            heur_no_accept_time, heur_no_accept_size, heur_no_accept_err, heur_no_accept_valid, heur_no_accept_status = self.execute_compression(
                filepath, heur_no_accept_7z_path, "heuristic", True, False, acc_float, acc_uint8, bs, sub_workers_str, rgb_ch
            )
            with self.results_lock:
                completed_steps += 1
                self.update_progress(completed_steps, total_steps)

            if heur_no_accept_status == "Success":
                heur_no_accept_ratio = orig_size / heur_no_accept_size if heur_no_accept_size > 0 else 0
                heur_no_accept_speed = orig_mb / heur_no_accept_time if heur_no_accept_time > 0 else 0
                self.log(f"  [{filename}] ✅ Heuristic (no accept) ratio: {heur_no_accept_ratio:.2f}x | Speed: {heur_no_accept_speed:.2f} MB/s | MaxErr: {heur_no_accept_err} | Valid: {heur_no_accept_valid}")
            else:
                self.log(f"  [{filename}] ❌ Heuristic (no accept) FAILED: {heur_no_accept_status}")
                heur_no_accept_ratio, heur_no_accept_speed = 0, 0

            # Mode 3: Advanced Heuristic WITH accept
            self.log(f"  [{filename}] Running mode: HEURISTIC (Advanced) WITH accept (using {sub_workers_str} processes)...")
            heur_accept_time, heur_accept_size, heur_accept_err, heur_accept_valid, heur_accept_status = self.execute_compression(
                filepath, heur_accept_7z_path, "heuristic", True, True, acc_float, acc_uint8, bs, sub_workers_str, rgb_ch
            )
            with self.results_lock:
                completed_steps += 1
                self.update_progress(completed_steps, total_steps)

            if heur_accept_status == "Success":
                heur_accept_ratio = orig_size / heur_accept_size if heur_accept_size > 0 else 0
                heur_accept_speed = orig_mb / heur_accept_time if heur_accept_time > 0 else 0
                self.log(f"  [{filename}] ✅ Heuristic (accept) ratio: {heur_accept_ratio:.2f}x | Speed: {heur_accept_speed:.2f} MB/s | MaxErr: {heur_accept_err} | Valid: {heur_accept_valid}")
            else:
                self.log(f"  [{filename}] ❌ Heuristic (accept) FAILED: {heur_accept_status}")
                heur_accept_ratio, heur_accept_speed = 0, 0
                
            # Clean up temp 7z archives if checked
            if delete_temp:
                for path in (bin_7z_path, heur_no_accept_7z_path, heur_accept_7z_path):
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception as e:
                            self.log(f"  [{filename}] ⚠ Failed to remove temporary file {os.path.basename(path)}: {e}")
                            
            # Store results and save CSV immediately after each file completes
            with self.results_lock:
                self.results_detailed.append({
                    "filename": filename, "mode": "binary", "orig_bytes": orig_size, "comp_bytes": bin_size,
                    "ratio": bin_ratio, "time_s": bin_time, "speed_mbs": bin_speed, "max_err": bin_err,
                    "valid": bin_valid, "status": bin_status
                })
                self.results_detailed.append({
                    "filename": filename, "mode": "heuristic_no_accept", "orig_bytes": orig_size, "comp_bytes": heur_no_accept_size,
                    "ratio": heur_no_accept_ratio, "time_s": heur_no_accept_time, "speed_mbs": heur_no_accept_speed, "max_err": heur_no_accept_err,
                    "valid": heur_no_accept_valid, "status": heur_no_accept_status
                })
                self.results_detailed.append({
                    "filename": filename, "mode": "heuristic_accept", "orig_bytes": orig_size, "comp_bytes": heur_accept_size,
                    "ratio": heur_accept_ratio, "time_s": heur_accept_time, "speed_mbs": heur_accept_speed, "max_err": heur_accept_err,
                    "valid": heur_accept_valid, "status": heur_accept_status
                })
                
                # Store comparison summary row
                if bin_status == "Success" and heur_no_accept_status == "Success" and heur_accept_status == "Success":
                    summary_row = {
                        "filename": filename,
                        "orig_mb": orig_mb,
                        "bin_size_mb": bin_size / (1024 * 1024),
                        "heur_no_accept_size_mb": heur_no_accept_size / (1024 * 1024),
                        "heur_accept_size_mb": heur_accept_size / (1024 * 1024),
                        "bin_ratio": bin_ratio,
                        "heur_no_accept_ratio": heur_no_accept_ratio,
                        "heur_accept_ratio": heur_accept_ratio,
                        "bin_time": bin_time,
                        "heur_no_accept_time": heur_no_accept_time,
                        "heur_accept_time": heur_accept_time,
                    }
                    self.results_summary.append(summary_row)

                    # Append row directly to UI Treeview in main thread
                    self.root.after(0, self.add_tree_row, summary_row)
                
                # SAVE CSV IMMEDIATELY TO DISK AFTER EACH ITERATION (avoid losing data on crash/interruption)
                self.save_csv_files(directory)

            self.log(f"--- [File {idx+1}/{len(tiff_files)}] Finished: {filename} or finished modes ---\n")

        # Run compilation using ThreadPoolExecutor with 5 parallel threads (processes 5 files at once)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_single_file, filepath, i) for i, filepath in enumerate(tiff_files)]
            concurrent.futures.wait(futures)
            
        # Render charts if Matplotlib is present and we have finished results
        if HAS_MATPLOTLIB and self.results_summary:
            self.root.after(0, self.render_charts, directory)
            
        self.root.after(0, self.benchmark_finished, "Success")
        
    def execute_compression(
        self,
        input_path: str,
        output_path: str,
        backend: str,
        adv_heur: bool,
        accept_pred: bool,
        acc_float: str,
        acc_uint8: str,
        block_size: str,
        workers: str,
        rgb_channel_indices: str = None
    ) -> Tuple[float, int, str, str, str]:
        """Runs compress.py in a subprocess and parses output logs."""
        cmd = [sys.executable, COMPRESS_SCRIPT, "-i", input_path, "-o", output_path, "--backend", backend]
        if adv_heur:
            cmd.append("--advanced-heuristic")
        else:
            cmd.append("--no-advanced-heuristic")

        if accept_pred:
            cmd.append("--accept-prediction")
        else:
            cmd.append("--no-accept-prediction")
            
        if block_size != "auto":
            cmd += ["--block-size", block_size]
        else:
            cmd.append("--auto")
            
        cmd += ["--accuracy", acc_float]
        cmd += ["--uint8-accuracy", acc_uint8]
        cmd += ["--uint8-accuracy-mode"]  # Force accuracy mode for uint8

        if rgb_channel_indices:
            cmd += ["--rgb-channel-indices", rgb_channel_indices]

        if workers != "auto":
            cmd += ["--processes", workers]
            
        t_start = time.time()
        try:
            self.log(f"    Executing: {' '.join(cmd)}")
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            t_end = time.time()
            elapsed = t_end - t_start
            
            # Read size of the compressed archive
            comp_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            
            # Parse logs for max_err and validity
            stdout_err = res.stdout + res.stderr
            max_err = "N/A"
            validity = "N/A"
            
            # Parse backend from subprocess output
            backend_match = re.search(r"Backend:\s+(\w+)", stdout_err)
            if backend_match:
                detected_backend = backend_match.group(1)
                self.log(f"    Detected backend: {detected_backend}")
            
            # Log first few lines of subprocess output for debugging
            lines = stdout_err.split('\n')[:10]
            if lines:
                self.log(f"    Subprocess output (first 10 lines):")
                for line in lines:
                    if line.strip():
                        self.log(f"      {line}")
            
            # Search for max_err and validity
            err_match = re.search(r"max_err[=\s]([0-9\.]+)", stdout_err)
            if not err_match:
                err_match = re.search(r"max_error[=\s]([0-9\.]+)", stdout_err)
                
            valid_match = re.search(r"valid[=\s](True|False|TRUE|FALSE)", stdout_err, re.IGNORECASE)
            if not valid_match:
                valid_match = re.search(r"validity[=\s](True|False|TRUE|FALSE)", stdout_err, re.IGNORECASE)
                
            if err_match:
                max_err = err_match.group(1)
            if valid_match:
                validity = valid_match.group(1).upper()
                
            return elapsed, comp_size, max_err, validity, "Success"
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - t_start
            err_msg = f"Process exited with code {e.returncode}. Output:\n{e.stdout}\n{e.stderr}"
            return elapsed, 0, "N/A", "N/A", err_msg
        except Exception as e:
            elapsed = time.time() - t_start
            return elapsed, 0, "N/A", "N/A", str(e)
            
    def update_progress(self, current: int, total: int):
        pct = (current / total) * 100
        self.root.after(0, lambda: self.progress_bar.configure(value=pct))
        
    def add_tree_row(self, row: Dict[str, Any]):
        self.tree.insert("", tk.END, values=(
            row["filename"],
            f"{row['orig_mb']:.2f}",
            f"{row['bin_ratio']:.2f}x",
            f"{row['heur_no_accept_ratio']:.2f}x",
            f"{row['heur_accept_ratio']:.2f}x",
            f"{row['bin_time']:.2f}",
            f"{row['heur_no_accept_time']:.2f}",
            f"{row['heur_accept_time']:.2f}"
        ))
        
    def save_csv_files(self, output_dir: str):
        # 1. Detailed CSV
        detailed_path = os.path.join(output_dir, "benchmark_detailed_results.csv")
        try:
            with open(detailed_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Filename", "Mode", "Original_Bytes", "Compressed_Bytes", 
                    "Compression_Ratio", "Time_Seconds", "Speed_MB_s", 
                    "Max_Error", "Validity", "Status"
                ])
                for r in self.results_detailed:
                    writer.writerow([
                        r["filename"], r["mode"], r["orig_bytes"], r["comp_bytes"],
                        f"{r['ratio']:.4f}", f"{r['time_s']:.4f}", f"{r['speed_mbs']:.4f}",
                        r["max_err"], r["valid"], r["status"]
                    ])
            self.log(f"💾 Saved detailed results: {detailed_path}")
        except Exception as e:
            self.log(f"❌ Failed to save detailed CSV: {e}")
            
        # 2. Summary/Comparison CSV
        summary_path = os.path.join(output_dir, "benchmark_comparison_summary.csv")
        try:
            with open(summary_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Filename", "Original_Size_MB", "Binary_Compressed_MB",
                    "Heur_No_Accept_Compressed_MB", "Heur_Accept_Compressed_MB",
                    "Binary_Ratio", "Heur_No_Accept_Ratio", "Heur_Accept_Ratio",
                    "Binary_Time_Seconds", "Heur_No_Accept_Time_Seconds", "Heur_Accept_Time_Seconds"
                ])
                for r in self.results_summary:
                    writer.writerow([
                        r["filename"], f"{r['orig_mb']:.4f}", f"{r['bin_size_mb']:.4f}",
                        f"{r['heur_no_accept_size_mb']:.4f}", f"{r['heur_accept_size_mb']:.4f}",
                        f"{r['bin_ratio']:.4f}", f"{r['heur_no_accept_ratio']:.4f}", f"{r['heur_accept_ratio']:.4f}",
                        f"{r['bin_time']:.4f}", f"{r['heur_no_accept_time']:.4f}", f"{r['heur_accept_time']:.4f}"
                    ])
            self.log(f"💾 Saved summary comparison: {summary_path}")
        except Exception as e:
            self.log(f"❌ Failed to save summary CSV: {e}")
            
    def render_charts(self, output_dir: str):
        if not self.results_summary:
            return
            
        # Extract fields
        filenames = [r["filename"] for r in self.results_summary]
        # Shorten filenames if they are too long
        short_names = [f[:15] + "..." if len(f) > 18 else f for f in filenames]

        bin_ratios = [r["bin_ratio"] for r in self.results_summary]
        heur_no_accept_ratios = [r["heur_no_accept_ratio"] for r in self.results_summary]
        heur_accept_ratios = [r["heur_accept_ratio"] for r in self.results_summary]

        bin_times = [r["bin_time"] for r in self.results_summary]
        heur_no_accept_times = [r["heur_no_accept_time"] for r in self.results_summary]
        heur_accept_times = [r["heur_accept_time"] for r in self.results_summary]

        # Set up matplotlib figure
        fig = Figure(figsize=(10, 6), dpi=100)

        # Subplot 1: Compression Ratio Comparison
        ax1 = fig.add_subplot(1, 2, 1)
        x = np.arange(len(filenames))
        width = 0.25

        ax1.bar(x - width, bin_ratios, width, label="Binary Search", color="#2c3e50")
        ax1.bar(x, heur_no_accept_ratios, width, label="Adv Heur (no accept)", color="#e74c3c")
        ax1.bar(x + width, heur_accept_ratios, width, label="Adv Heur (accept)", color="#f39c12")
        ax1.set_ylabel("Compression Ratio (higher is better)")
        ax1.set_title("Compression Ratio Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Subplot 2: Compression Time (Speedup) Comparison
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.bar(x - width, bin_times, width, label="Binary Search", color="#34495e")
        ax2.bar(x, heur_no_accept_times, width, label="Adv Heur (no accept)", color="#e67e22")
        ax2.bar(x + width, heur_accept_times, width, label="Adv Heur (accept)", color="#f1c40f")
        ax2.set_ylabel("Compression Time in Seconds (lower is better)")
        ax2.set_title("Compression Speed Comparison")
        ax2.set_xticks(x)
        ax2.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)
        
        fig.tight_layout()
        
        # Save PNG to disk
        chart_path = os.path.join(output_dir, "benchmark_comparison_charts.png")
        try:
            fig.savefig(chart_path)
            self.log(f"📊 Saved comparison chart image: {chart_path}")
        except Exception as e:
            self.log(f"⚠ Failed to save comparison charts image: {e}")
            
        # Draw on GUI Canvas Tab
        # Clear previous canvas if any
        if self.charts_canvas:
            self.charts_canvas.get_tk_widget().destroy()
            
        self.charts_canvas = FigureCanvasTkAgg(fig, master=self.charts_tab)
        self.charts_canvas.draw()
        self.charts_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def benchmark_finished(self, status: str):
        self.is_running = False
        self.btn_run.config(state="normal")
        if status == "Success":
            self.status_lbl.config(text="Status: Completed successfully!")
            self.log("\n✨ BENCHMARKS COMPLETED SUCCESSFULLY! ✨")
            self.log("Check the tables and charts tabs to analyze the results.")
            messagebox.showinfo("Benchmark Completed", "The batch benchmark run finished successfully!\nDetailed CSV files and plots have been saved.")
        else:
            self.status_lbl.config(text=f"Status: Error - {status}")
            self.log(f"\n❌ BENCHMARK ABORTED: {status}")
            messagebox.showerror("Benchmark Failed", f"Benchmark aborted due to error:\n{status}")


def main():
    root = tk.Tk()
    app = BenchmarkGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
