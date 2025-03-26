import streamlit as st
import json
import os
import subprocess

st.set_page_config(
    page_title="FIDWaC",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.help.com',
        'Report a bug': "https://www.bug.com",
        'About': (
            "**FIDWaC (Fast Inverse Distance Weighting and Compression)** is a Python toolkit for:\n\n"
            "- **Fast creation of continuous surfaces** (rasters) from scattered measurement points using the IDW (Inverse Distance Weighting) method\n"
            "- **Lossy compression** of raster data (GeoTIFF) using **Discrete Cosine Transform (DCT)** and zigzag encoding\n\n"
            "The package is designed for efficient processing and storage of geospatial data in fields such as:\n\n"
            "- Bathymetry analysis\n"
            "- Digital terrain modeling\n"
            "- Shoreline monitoring\n"
            "- Hydrology, climatology, and other applications requiring large spatial datasets"
        )
    }
)

def load_css():
    css_path = "./style/style.css"
    with open(css_path, "r") as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Load CSS
load_css()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTERPOLATION_PATH = os.path.join(BASE_DIR, "interpolation")
COMPRESSION_PATH = os.path.join(BASE_DIR, "compression")

# Paths to config files
INTERPOLATION_CONFIG_FILE = os.path.join("interpolation", "config.json")
COMPRESSION_CONFIG_FILE = os.path.join("compression", "config.json")

# Function to load JSON config
def load_config(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        st.warning("Configuration file does not exist. A new one has been created.")
        return {}

# Function to save JSON config
def save_config(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

# Correct relative paths in interpolation config
def fix_interpolation_paths(config):
    if config.get("results_directory", "").startswith("./"):
        config["results_directory"] = os.path.join(INTERPOLATION_PATH, config["results_directory"][2:])
    if config.get("geoid_correction_file", "").startswith("./"):
        config["geoid_correction_file"] = os.path.join(INTERPOLATION_PATH, config["geoid_correction_file"][2:])
    return config

# Correct relative paths in compression config
def fix_compression_paths(config):
    if config.get("results_directory", "").startswith("./"):
        config["results_directory"] = os.path.join(COMPRESSION_PATH, config["results_directory"][2:])
    if config.get("source_directory", "").startswith("./"):
        config["source_directory"] = os.path.join(COMPRESSION_PATH, config["source_directory"][2:])
    return config

# Sidebar Navigation
with st.sidebar:
    st.title("FIDWaC")
    st.caption("Fast IDW & Compression Toolkit")

# Główna sekcja (Interpolation / Compression)
main_section = st.sidebar.selectbox("Select Section", ["Interpolation", "Compression"])

if main_section == "Interpolation":
    page = st.sidebar.radio("Interpolation", ["Config", "Script"])
elif main_section == "Compression":
    page = st.sidebar.radio("Compression", ["Config", "Script"])

# ---------------------- Interpolation Config ----------------------
if main_section == "Interpolation" and page == "Config":
    st.title("IDW Interpolation - Configuration")
    st.write("Configuration settings for IDW interpolation.")

    config = fix_interpolation_paths(load_config(INTERPOLATION_CONFIG_FILE))

    st.subheader("General Settings")
    config["results_directory"] = st.text_input("Results Directory", config.get("results_directory", "./results/"),
                                                help="The directory where all output results will be saved.")
    config["z_field_name"] = st.text_input("Z Field Name", config.get("z_field_name", "z"),
                                           help="The attribute field name representing elevation or intensity values.")

    st.subheader("LAS Filtering")
    config["filter_las"] = st.checkbox("Enable LAS Filtering", config.get("filter_las", True),
                                       help="Enable filtering of LAS files based on classification.")
    config["filter_classes"] = st.multiselect("Allowed LAS Classes (e.g., 2 = Ground)",
                                             list(range(0, 21)), default=config.get("filter_classes", [2]),
                                             help="Specify which LAS classification classes should be used for processing.")

    st.subheader("Interpolation Parameters")
    config["N"] = st.number_input("Number of Neighboring Points (N)", min_value=1, max_value=100, value=config.get("N", 15),
                                  help="Defines how many closest points will be used for interpolation.")
    config["resolution"] = st.number_input("Resolution (Grid Cell Size)", min_value=0.01, step=0.01, value=config.get("resolution", 0.1),
                                           help="Defines the pixel size of the output raster.")
    config["max_distance"] = st.number_input("Maximum Search Distance", value=config.get("max_distance", 15),
                                             help="Defines the maximum distance within which points are considered for interpolation.")
    config["leafsize"] = st.number_input("Leaf Size (for KD-Tree)", value=config.get("leafsize", 50),
                                         help="Defines the number of points stored per node in the KD-tree structure.")
    config["weights"] = st.slider("IDW Weighting Factor", min_value=0.0, max_value=2.0, value=float(config.get("weights", 0.8)), step=0.1,
                        help="Controls how much influence nearby points have on the interpolation.\nHigher values give more weight to close points, reducing smoothing."
)

    st.subheader("Smoothing Options")
    config["smooth_result"] = st.checkbox(
        "Apply Smoothing to Interpolated Output",
        config.get("smooth_result", True),
        help="Enable this option to apply a smoothing filter to the interpolated raster surface."
    )
    config["smooth_level"] = st.slider(
        "Smoothing Level (Kernel Size)",
        min_value=1, max_value=5, step=1,
        value=config.get("smooth_level", 1),
        help="Controls how aggressively the raster is smoothed.\nHigher values apply stronger smoothing."
    )

    st.subheader("Geoid Correction")
    config["geoid_correction"] = st.checkbox(
        "Enable Geoid Correction",
        config.get("geoid_correction", True),
        help="Enable vertical correction based on geoid elevation data."
    )
    config["geoid_correction_file"] = st.text_input(
        "Geoid Correction File Path",
        config.get("geoid_correction_file", "./geoid/PL_geod.csv"),
        help="Path to the CSV file containing geoid correction values."
    )
    config["geoidCrs"] = st.text_input(
        "Geoid CRS",
        config.get("geoidCrs", "epsg:4326"),
        help="CRS used by the geoid correction file (typically EPSG:4326 for lat/lon)."
    )

    st.subheader("Additional Settings")
    config["sourceCrs"] = st.text_input(
        "Source CRS (Coordinate Reference System)",
        config.get("sourceCrs", "epsg:2180"),
        help="CRS of the source input data (e.g., EPSG code)."
    )
    st.subheader("Output Formats")
    st.warning(
        "⚠️ Using CSV, TXT, or SHP formats with large datasets may significantly increase processing time. "
        "Consider this when exporting final results."
    )
    config["save_data_to_shp"] = st.checkbox(
        "Save results as SHP",
        config.get("save_data_to_shp", False),
        help="Export interpolated results as a Shapefile (.shp). May be slow on large datasets."
    )
    config["csv_result"] = st.checkbox(
        "Save results as CSV",
        config.get("csv_result", False),
        help="Export interpolated results as a CSV file. Large files may take time to generate."
    )
    config["surfer_grd"] = st.checkbox(
        "Save as Surfer Grid (.grd)",
        config.get("surfer_grd", False),
        help="Save output in Golden Software Surfer Grid format. Suitable for use with Surfer."
    )
    config["interpolation_image_create"] = st.checkbox(
        "Generate Interpolation Images",
        config.get("interpolation_image_create", True),
        help="Create raster images showing the interpolation results for visual analysis."
    )

    st.subheader("Interpolation Methods")
    method = st.radio(
        "Choose one interpolation method:",
        ("IDW with Dask", "IDW with NumPy", "KNN Estimation"),
        index=2
    )

    # Help text for each method
    if method == "IDW with Dask":
        st.caption("⚙️ Uses Dask for parallel IDW interpolation.")
    elif method == "IDW with NumPy":
        st.caption("⚡ Performs fast IDW interpolation using NumPy.")
    elif method == "KNN Estimation":
        st.caption("📊 Uses k-Nearest Neighbors for estimation.")

    # Logical flags for later use
    config["idw_dask"] = method == "IDW with Dask"
    config["idw_numpy"] = method == "IDW with NumPy"
    config["knn_calculate"] = method == "KNN Estimation"

    if st.button("Save Configuration"):
        save_config(config, INTERPOLATION_CONFIG_FILE)
        st.success("Configuration saved successfully.")

    st.subheader("Configuration Preview")
    st.json(config)

# ---------------------- Interpolation Script ----------------------
if "run_folder_path" not in st.session_state:
    st.session_state["run_folder_path"] = ""

if main_section == "Interpolation" and page == "Script":
    st.title("IDW Interpolation - Run Script")
    st.write("Run the interpolation script with selected parameters.")

    # File Selection
    st.subheader("Select LAS/LAZ/SHP/TXT File or Directory")

    file_mode = st.radio("Choose input type:", ["Single File", "Batch Processing (Folder)"])

    file_path = ""
    num_threads = 1

    if file_mode == "Single File":
        file_path = st.text_input("Enter file path", "interpolation/source/sample.las")
        if file_path and os.path.isfile(file_path):
            st.success(f"File found: `{file_path}` ✅")
        elif file_path:
            st.error(f"File not found: `{file_path}` ❌")

    else:
        if "batch_folder_path" not in st.session_state:
            st.session_state["batch_folder_path"] = ""

        st.session_state["batch_folder_path"] = st.text_input(
            "Enter folder path for batch processing (e.g., ./lidar)",
            value=st.session_state["batch_folder_path"]
        )

        folder_path = st.session_state["batch_folder_path"]

        if folder_path and os.path.isdir(folder_path):
            st.success(f"Folder found: `{folder_path}` ✅")
            valid_exts = (".las", ".laz", ".shp", ".txt")
            las_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]

            if las_files:
                st.subheader("Files in selected folder:")
                st.write(" ".join(las_files))
            else:
                st.warning(f"No supported files (.las/.laz/.shp/.txt) found in `{folder_path}` ❌")

            num_threads = st.slider("Number of parallel threads", min_value=1, max_value=12, value=1,
                                    help="Defines the number of parallel processes for batch processing. NOTE: This process requires a lot of RAM, please be careful with a higher number of threads. ")
        elif folder_path:
            st.error(f"Folder not found: `{folder_path}` ❌")

    # Output log window
    log_area = st.empty()

    # Run Script
    if st.button("Run Script"):
        cmd = None

        if file_mode == "Single File" and file_path and os.path.isfile(file_path):
            cmd = f'python interpolation_FIT.py "{file_path}"'
            st.code(cmd, language="bash")

        elif file_mode.startswith("Batch Processing") and os.path.isdir(st.session_state["batch_folder_path"]):
            folder_path = st.session_state["batch_folder_path"]
            abs_path = os.path.abspath(folder_path)

            valid_exts = (".las", ".laz", ".shp", ".txt")
            input_files = [f for f in os.listdir(abs_path) if f.lower().endswith(valid_exts)]

            st.write("Input files =", input_files)

            if not input_files:
                st.error(f"No valid input files found in `{abs_path}` ❌")
            else:
                input_files_full = [os.path.join(abs_path, f) for f in input_files]
                input_str = ' '.join(f'"{f}"' for f in input_files_full)
                cmd = f'parallel -j {num_threads} python -u interpolation_FIT.py ::: {input_str}'
                #st.code(f"Executing batch command:\n{cmd}", language="bash")

        else:
            st.warning("Please enter a valid file or folder path.")

        # --- Processing ---
        if cmd:
            with st.spinner("Processing..."):
                st.write("Final command to execute:")
                st.code(cmd, language="bash")

                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

                output_lines = []
                for line in iter(process.stdout.readline, ''):
                    output_lines.append(line.strip())
                    log_area.text("\n".join(output_lines))

                process.stdout.close()
                process.wait()

                error_output = process.stderr.read().strip()
                if error_output:
                    st.subheader("Errors ❌")
                    st.text(error_output)

            st.success("Processing Complete ✅")



# ---------------------- Compression Config ----------------------
elif main_section == "Compression" and page == "Config":
    st.title("Compression - Configuration")
    st.write("Configure parameters for raster compression using DCT.")

    config = fix_compression_paths(load_config(COMPRESSION_CONFIG_FILE))

    st.subheader("Directories")
    config["source_directory"] = st.text_input("Source Directory", config.get("source_directory", "./source/"),
                                               help="Directory containing input raster (GeoTIFF) files.")
    config["results_directory"] = st.text_input("Results Directory", config.get("results_directory", "./results/"),
                                                help="Directory where compressed outputs will be saved.")

    st.subheader("Compression Parameters")
    config["accuracy"] = st.number_input("Compression Accuracy", min_value=0.0, max_value=1.0, step=0.01,
                                         value=config.get("accuracy", 0.05),
                                         help="Target maximum average error (e.g., 0.05 for 5cm).")
    config["matrix"] = st.selectbox("Matrix Size for DCT", [8, 16, 32, 64],
                                    index=[8, 16, 32, 64].index(config.get("matrix", 16)),
                                    help="Size of the matrix used for DCT transformation.")
    config["decimal"] = st.number_input("Decimal Precision", min_value=0, max_value=6,
                                        value=config.get("decimal", 2),
                                        help="Number of decimal places to round the data.")
    config["type_dct"] = st.selectbox("DCT Type", [1, 2, 3], index=[1, 2, 3].index(config.get("type_dct", 2)),
                                      help="Type of Discrete Cosine Transform used.")
    config["sourceCrs_force_declare"] = st.checkbox(
        "Force CRS Declaration",
        value=config.get("sourceCrs_force_declare", False),
        help="Force declaration of source CRS when not included in the source file (e.g. ASCII grid)."
    )

    if config["sourceCrs_force_declare"]:
        config["sourceCrs"] = st.text_input(
            "Enter Source CRS (e.g., EPSG:2180)",
            value=config.get("sourceCrs", "EPSG:2180"),
            help="Specify the Coordinate Reference System (e.g. EPSG:2180)."
        )

    if st.button("Save Configuration"):
        save_config(config, COMPRESSION_CONFIG_FILE)
        st.success("Configuration saved successfully.")

    st.subheader("Preview Configuration")
    st.json(config)


# ---------------------- Compression Script ----------------------
elif main_section == "Compression" and page == "Script":
    st.title("Compression - Run Script")
    st.write("Run the compression script using selected configuration parameters.")

    # File/Folder Selection
    st.subheader("Select TIFF File or Directory")
    comp_mode = st.radio("Choose input type:", ["Single File", "Batch Processing (Folder)"])

    comp_file_path = ""
    comp_folder_path = ""
    comp_threads = 1
    valid_exts = (".tif", ".tiff")

    if comp_mode == "Single File":
        comp_file_path = st.text_input("Enter GeoTIFF file path", "compression/source/sample.tif")
        if comp_file_path and os.path.isfile(comp_file_path):
            st.success(f"File found: `{comp_file_path}` ✅")
        elif comp_file_path:
            st.error(f"File not found: `{comp_file_path}` ❌")
    else:
        if "comp_folder_path" not in st.session_state:
            st.session_state["comp_folder_path"] = ""

        st.session_state["comp_folder_path"] = st.text_input(
            "Enter folder path for batch compression",
            value=st.session_state["comp_folder_path"]
        )

        comp_folder_path = st.session_state["comp_folder_path"]

        if comp_folder_path and os.path.isdir(comp_folder_path):
            st.success(f"Folder found: `{comp_folder_path}` ✅")
            tiff_files = [f for f in os.listdir(comp_folder_path) if f.lower().endswith(valid_exts)]

            if tiff_files:
                st.subheader("Files in selected folder:")
                st.write(" ".join(tiff_files))
            else:
                st.warning(f"No .tif/.tiff files found in `{comp_folder_path}` ❌")

            comp_threads = st.slider("Number of parallel threads", min_value=1, max_value=12, value=2,
                            help="Defines the number of parallel processes for batch processing. NOTE: This process requires a lot of RAM, please be careful with a higher number of threads. ")

        elif comp_folder_path:
            st.error(f"Folder not found: `{comp_folder_path}` ❌")

    log_area = st.empty()

    if st.button("Run Compression"):
        cmd = None
        compress_script = os.path.join("compression", "compress.py")

        if comp_mode == "Single File" and os.path.isfile(comp_file_path):
            relative_input = os.path.relpath(comp_file_path, start="compression")
            cmd = f'cd compression && python compress.py "{relative_input}" && cd ..'

        elif comp_mode.startswith("Batch") and os.path.isdir(comp_folder_path):
            abs_path = os.path.abspath(comp_folder_path)
            input_files = [os.path.join(abs_path, f) for f in os.listdir(abs_path) if f.lower().endswith(valid_exts)]

            if not input_files:
                st.error("No valid input files to compress.")
            else:
                input_str = ' '.join(f'"{os.path.relpath(f, "compression")}"' for f in input_files)
                cmd = f'cd compression && parallel -j {comp_threads} python compress.py ::: {input_str} && cd ..'


        else:
            st.warning("Please enter a valid file or folder path.")

        if cmd:
            with st.spinner("Processing..."):
                st.code(cmd, language="bash")
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

                output_lines = []
                for line in iter(process.stdout.readline, ''):
                    output_lines.append(line.strip())
                    log_area.text("\n".join(output_lines))

                process.stdout.close()
                process.wait()

                error_output = process.stderr.read().strip()
                if error_output and "Traceback" in error_output:
                    st.subheader("Errors ❌")
                    st.text(error_output)
                elif error_output:
                    st.subheader("Log Output (stderr)")
                    st.text(error_output)


            st.success("Compression Complete ✅")
