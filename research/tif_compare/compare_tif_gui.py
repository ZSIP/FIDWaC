#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import rasterio
import numpy as np
import threading
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class TiffComparerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TIFF Comparator - GeoTIFF File Comparison Tool")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)

        self.file1 = tk.StringVar()
        self.file2 = tk.StringVar()
        self.status = tk.StringVar(value="Ready")

        self.setup_ui()

    def setup_ui(self):
        # Header frame
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(header_frame, text="TIFF Comparator", font=("Arial", 16, "bold")).pack()
        ttk.Label(header_frame, text="Comparison and difference analysis of GeoTIFF files").pack()

        # File selection frame
        file_frame = ttk.LabelFrame(self.root, text="File Selection", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=10)

        # File 1
        ttk.Label(file_frame, text="File 1:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.file1, width=60).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file1).grid(row=0, column=2)

        # File 2
        ttk.Label(file_frame, text="File 2:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.file2, width=60).grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file2).grid(row=1, column=2)

        # Action buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="Compare Files", command=self.compare_files).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(button_frame, text="Clear", command=self.clear_results).pack(
            side=tk.LEFT, padx=5
        )

        # Status bar
        ttk.Label(self.root, textvariable=self.status, relief=tk.SUNKEN).pack(
            fill=tk.X, side=tk.BOTTOM
        )

        # Notebook with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: Statistics
        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="Statistics")
        self.setup_stats_tab()

        # Tab 2: Charts
        self.charts_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.charts_frame, text="Charts")

    def setup_stats_tab(self):
        canvas = tk.Canvas(self.stats_frame)
        scrollbar = ttk.Scrollbar(self.stats_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        self.stats_text = tk.Text(
            scrollable_frame, width=120, height=30, wrap=tk.WORD, font=("Courier", 10)
        )
        self.stats_text.pack()

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def browse_file1(self):
        file_path = filedialog.askopenfilename(
            title="Select first TIFF file",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
        )
        if file_path:
            self.file1.set(file_path)

    def browse_file2(self):
        file_path = filedialog.askopenfilename(
            title="Select second TIFF file",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
        )
        if file_path:
            self.file2.set(file_path)

    def compare_files(self):
        if not self.file1.get() or not self.file2.get():
            messagebox.showerror("Error", "Select both files!")
            return

        self.status.set("Comparing files...")
        self.root.update()

        # Run in separate thread to avoid freezing GUI
        thread = threading.Thread(target=self._compare_worker)
        thread.start()

    def _compare_worker(self):
        try:
            file1 = self.file1.get()
            file2 = self.file2.get()

            # Reading files
            with rasterio.open(file1) as src1:
                data1 = src1.read()
                profile1 = src1.profile

            with rasterio.open(file2) as src2:
                data2 = src2.read()
                profile2 = src2.profile

            # Prepare data
            if data1.shape != data2.shape:
                min_rows = min(data1.shape[1], data2.shape[1])
                min_cols = min(data1.shape[2], data2.shape[2])
                min_bands = min(data1.shape[0], data2.shape[0])
                data1_crop = data1[:min_bands, :min_rows, :min_cols]
                data2_crop = data2[:min_bands, :min_rows, :min_cols]
                shape_warning = f"\n⚠️  WARNING: Different shapes! File1: {data1.shape}, File2: {data2.shape}\n    Used range: {data1_crop.shape}\n"
            else:
                data1_crop = data1
                data2_crop = data2
                shape_warning = ""

            # Compute difference
            difference = data1_crop.astype(np.float64) - data2_crop.astype(np.float64)
            abs_diff = np.abs(difference)

            # Statistics
            max_diff = np.nanmax(abs_diff)
            min_diff = np.nanmin(abs_diff)
            mean_diff = np.nanmean(abs_diff)
            std_diff = np.nanstd(abs_diff)
            median_diff = np.nanmedian(abs_diff)

            non_zero = np.sum(abs_diff > 0)
            total = abs_diff.size

            # Prepare statistics text
            stats_text = f"""
╔════════════════════════════════════════════════════════════════╗
║                     TIFF FILE COMPARISON                       ║
╚════════════════════════════════════════════════════════════════╝

📋 FILE INFORMATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File 1: {file1}
  Shape: {data1.shape}
  Data type: {data1.dtype}
  Min: {np.nanmin(data1):.10f}
  Max: {np.nanmax(data1):.10f}

File 2: {file2}
  Shape: {data2.shape}
  Data type: {data2.dtype}
  Min: {np.nanmin(data2):.10f}
  Max: {np.nanmax(data2):.10f}

{shape_warning}

📊 DIFFERENCE STATISTICS (file1 - file2):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  🔴 Max difference:         {max_diff:.10f}
  🟢 Min difference:         {min_diff:.10f}
  🟡 Mean difference:        {mean_diff:.10f}
  📈 Median difference:      {median_diff:.10f}
  📉 Std deviation:          {std_diff:.10f}
  
  Pixels with diff > 0:     {non_zero:,} / {total:,} ({100*non_zero/total:.2f}%)

📈 DIFFERENCE PERCENTILES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                val = np.nanpercentile(abs_diff, p)
                stats_text += f"  Percentile {p:3d}%:  {val:.10f}\n"

            stats_text += "\n"

            # Update UI
            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, stats_text)
            self.stats_text.config(state=tk.DISABLED)

            # Update charts
            self.update_charts(abs_diff, data1_crop, data2_crop)

            self.status.set(f"✅ Comparison complete. Max difference: {max_diff:.10f}")

        except Exception as e:
            messagebox.showerror("Error", f"Error during comparison:\n{str(e)}")
            self.status.set("❌ Error!")

    def update_charts(self, abs_diff, data1, data2):
        # Clear previous charts
        for widget in self.charts_frame.winfo_children():
            widget.destroy()

        # Create figure with multiple subplots
        fig = Figure(figsize=(12, 10), dpi=100)

        # Difference histogram
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.hist(abs_diff.flatten(), bins=100, color="steelblue", edgecolor="black", alpha=0.7)
        ax1.set_xlabel("Difference value")
        ax1.set_ylabel("Pixel count")
        ax1.set_title("Difference histogram (absolute values)")
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.boxplot([abs_diff.flatten()], labels=["Differences"])
        ax2.set_ylabel("Difference value")
        ax2.set_title("Difference box plot")
        ax2.grid(True, alpha=0.3)

        # Text statistics
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.axis("off")
        stats_text = f"""Statistics:
Max: {np.nanmax(abs_diff):.10f}
Min: {np.nanmin(abs_diff):.10f}
Mean: {np.nanmean(abs_diff):.10f}
Median: {np.nanmedian(abs_diff):.10f}
Std: {np.nanstd(abs_diff):.10f}
Q1: {np.nanpercentile(abs_diff, 25):.10f}
Q3: {np.nanpercentile(abs_diff, 75):.10f}"""
        ax3.text(
            0.1,
            0.5,
            stats_text,
            fontsize=11,
            verticalalignment="center",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Min/Max/Mean value comparison across files
        ax4 = fig.add_subplot(2, 2, 4)
        categories = ["Min", "Max", "Mean"]
        file1_vals = [np.nanmin(data1), np.nanmax(data1), np.nanmean(data1)]
        file2_vals = [np.nanmin(data2), np.nanmax(data2), np.nanmean(data2)]

        x = np.arange(len(categories))
        width = 0.35

        ax4.bar(x - width / 2, file1_vals, width, label="File 1", alpha=0.8)
        ax4.bar(x + width / 2, file2_vals, width, label="File 2", alpha=0.8)
        ax4.set_ylabel("Value")
        ax4.set_title("Min/Max/Mean value comparison")
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()

        # Embed chart in GUI
        canvas = FigureCanvasTkAgg(fig, master=self.charts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def clear_results(self):
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.config(state=tk.DISABLED)
        self.status.set("Ready")
        for widget in self.charts_frame.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TiffComparerGUI(root)
    root.mainloop()
