# FIDWaC v2

Lossy block-DCT compression of rasters with a guaranteed maximum error: float32 elevation models, uint8 / uint16 bands and RGB orthophotos. Archives are `.7z` with msgpack metadata; georeferencing and NoData are stored in the archive.

**Documentation**

- [README_EN.md](README_EN.md) - full documentation

## Quick start

```bash
bash install/setup_linux.sh      # Linux / WSL2, see install/README.md
bash run_linux.sh                # web GUI at http://127.0.0.1:5010/
python app_tk.py                 # Tkinter desktop GUI
python compress.py --help        # command line
python interpolate.py --help     # IDW interpolation of scattered points
```

## Data

The FIDWAC dataset (DSM, DTM, multiband) is on Zenodo, DOI [10.5281/zenodo.21393089](https://doi.org/10.5281/zenodo.21393089).

## License

MIT, see [LICENSE](LICENSE).
