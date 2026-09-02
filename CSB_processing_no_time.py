"""
Deep-Water Optimized CSB Processing Script (no-time)
Asymmetric reference threshold check only

Inputs:
  - CSB CSV file  (columns: lat, lon, depth; unique_id/platform_name optional)
  - Reference bathymetry GeoTIFF  (with negative depth values)
  - Optional TID GeoTIFF          (restrict to only direct-measurement cells)

Output variables:
    - depth: identical to provided depth but always negative for usability within larger bathymetry grid
    - raster_val: sampled reference bathymetry (negative)
    - has_reference: boolean indicating if raster_val is valid
    - tid_value: sampled TID value (if provided)
    - outlier: boolean indicating if point was flagged as an outlier
    - outlier_reason: string indicating which layer flagged the point
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import tkinter as tk
from tkinter import filedialog, messagebox

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Tunable parameters (see CSB Processing SOP for details, section references below)

# Asymmetric reference thresolds (4.II.B.2)
# Dir. measurement cells
REF_FRAC_DIRECT = 0.10
REF_MIN_M_DIRECT = 5.0
REF_CAP_DIRECT = 100.0

# Ind. measurement cells
REF_FRAC_INDIRECT = 0.30
REF_MIN_M_INDIRECT = 20.0
REF_CAP_INDIRECT_DEEPER = 150.0     # Tight cap if CSB is deeper (likely bottom loss/noise)
REF_CAP_INDIRECT_SHALLOWER = 800.0  # Loose cap if CSB is shallower (allows for uncharted seamounts)

DIRECT_TID_VALUES = set(range(10, 18)) # 4.II.B.5.b


def pick_paths_via_gui() -> dict:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update_idletasks()

    print("Opening file manager - check terminal for prompt")
    print("Select folder containing CSB CSV files")
    csv_folder = filedialog.askdirectory(title="Select folder containing CSB CSV files")
    if not csv_folder: raise SystemExit("No CSV folder selected — exiting.")

    print("Select reference bathymetry GeoTIFF")
    raster_path = filedialog.askopenfilename(
        title="Select reference bathymetry GeoTIFF",
        filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")]
    )
    if not raster_path: raise SystemExit("No reference raster selected — exiting.")

    use_tid = messagebox.askyesno("TID Filter", "Do you have a TID GeoTIFF for direct measurements?")
    tid_path = None
    if use_tid:
        print("Select TID GeoTIFF")
        tid_path = filedialog.askopenfilename(
            title="Select TID GeoTIFF", filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")]
        )

    print("Select output folder")
    out_dir = filedialog.askdirectory(title="Select output folder")
    if not out_dir: raise SystemExit("No output folder selected — exiting.")

    root.destroy()
    return dict(csv_folder=csv_folder, raster_path=raster_path, tid_path=tid_path, out_dir=out_dir)


def _parse_single_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    required = {"lat", "lon", "depth"}
    if required - set(df.columns):
        raise ValueError(f"{os.path.basename(csv_path)}: missing columns")

    if "unique_id" not in df.columns: df["unique_id"] = "unknown_vessel"
    if "platform_name" not in df.columns: df["platform_name"] = "unknown"

    df["depth"] = pd.to_numeric(df["depth"], errors="coerce").abs()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    df = df.dropna(subset=["depth", "lat", "lon", "unique_id"])
    df = df[(df["depth"] > 0.5) & (df["depth"] < 11000)]
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
    return df.drop_duplicates(subset=["lon", "lat", "depth", "unique_id"])

def load_csb(csv_folder: str) -> gpd.GeoDataFrame:
    csv_files = sorted(p for p in Path(csv_folder).iterdir() if p.suffix.lower() == ".csv")
    frames = [_parse_single_csv(str(p)) for p in csv_files]
    df = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")

# Raster sampling

def sample_raster(gdf: gpd.GeoDataFrame, raster_path: str, col_name: str) -> gpd.GeoDataFrame:
    coords = [(geom.x, geom.y) for geom in gdf.geometry]
    with rasterio.open(raster_path) as src:
        sampled = np.array([v[0] for v in src.sample(coords)], dtype=float)
        if src.nodata is not None:
            sampled[np.isclose(sampled, src.nodata)] = np.nan
    sampled[sampled == 0] = np.nan
    gdf[col_name] = np.abs(sampled)
    return gdf

# Outlier Detection

def flag_asymmetric_ref(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Asymmetric reference check based on TID confidence."""
    has_tid = "tid_value" in gdf.columns
    mask_ref = gdf["has_reference"]

    if has_tid:
        direct_mask = mask_ref & gdf["tid_value"].apply(lambda t: pd.notna(t) and int(t) in DIRECT_TID_VALUES)
        indirect_mask = mask_ref & ~direct_mask
    else:
        direct_mask = pd.Series(False, index=gdf.index)
        indirect_mask = mask_ref

    new_flags = 0

    # Direct (Symmetric)
    if direct_mask.any():
        dev = np.abs(gdf.loc[direct_mask, "depth"] - gdf.loc[direct_mask, "raster_val"])
        thr = np.maximum(REF_MIN_M_DIRECT, gdf.loc[direct_mask, "depth"] * REF_FRAC_DIRECT)
        flag = (dev > thr) | (dev > REF_CAP_DIRECT)
        idx = gdf.loc[direct_mask][flag].index
        gdf.loc[idx, "outlier"] = True
        gdf.loc[idx, "outlier_reason"] = "hard_ref_direct"
        new_flags += len(idx)

    # Indirect (Asymmetric)
    if indirect_mask.any():
        sub = gdf.loc[indirect_mask]
        is_deeper = sub["depth"] > sub["raster_val"]
        dev = np.abs(sub["depth"] - sub["raster_val"])
        thr = np.maximum(REF_MIN_M_INDIRECT, sub["depth"] * REF_FRAC_INDIRECT)

        flag_deeper = is_deeper & ((dev > thr) | (dev > REF_CAP_INDIRECT_DEEPER))
        flag_shallower = (~is_deeper) & ((dev > thr) | (dev > REF_CAP_INDIRECT_SHALLOWER))

        idx = sub[flag_deeper | flag_shallower].index
        gdf.loc[idx, "outlier"] = True
        gdf.loc[idx, "outlier_reason"] = "hard_ref_indirect_asymmetric"
        new_flags += len(idx)

    print(f"    Asymmetric Ref Check: {new_flags:,} points flagged")
    return gdf


def main():
    paths = pick_paths_via_gui()
    OUT_DIR = paths["out_dir"]
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1/3] Loading Data...")
    gdf = load_csb(paths["csv_folder"])

    print("[2/3] Sampling Rasters...")
    gdf = sample_raster(gdf, paths["raster_path"], "raster_val")
    gdf["has_reference"] = gdf["raster_val"].notna()
    if paths["tid_path"]: gdf = sample_raster(gdf, paths["tid_path"], "tid_value")

    print("[3/3] Outlier Detection...")
    gdf["outlier"] = False
    gdf["outlier_reason"] = ""
    gdf = flag_asymmetric_ref(gdf)

    print("\nExporting...")

    # Export GPKG
    out = gdf.copy()
    out["depth"] = -out["depth"]
    out["raster_val"] = -out["raster_val"]
    out.to_file(os.path.join(OUT_DIR, "csb_processed.gpkg"), driver="GPKG")

    # Export CSVs (same attributes as GPKG, minus geometry)
    out_csv = out.drop(columns="geometry")
    out_csv.to_csv(os.path.join(OUT_DIR, "csb_processed.csv"), index=False)
    out_csv[~out_csv["outlier"]].to_csv(os.path.join(OUT_DIR, "csb_processed_no_outliers.csv"), index=False)

    print("Processing Complete.")

if __name__ == "__main__":
    main()