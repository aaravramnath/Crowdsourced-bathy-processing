"""
Deep-Water Optimized CSB Processing Script
Incorporates physical geometry limits, asymmetric reference thresholds, 
biological scattering layer detection, and Savitzky-Golay smoothing.

Inputs:
  - CSB CSV file  (columns: unique_id, platform_name, time, lat, lon, depth)
  - Reference bathymetry GeoTIFF  (with negative depth values)
  - Optional TID GeoTIFF          (restrict to only direct-measurement cells)

Output variables:
    - depth: identical to provided depth but always negative for usability within larger bathymetry grid
    - raster_val: sampled reference bathymetry (negative)
    - has_reference: boolean indicating if raster_val is valid
    - tid_value: sampled TID value (if provided)
    - transit_id: unique identifier for each continuous transit segment (represented in output plots)
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import tkinter as tk
from tkinter import filedialog, messagebox

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Tunable parameters (see CSB Processing SOP for details, section references below)

# Transit splitting (4.II.B.5.a)
TRANSIT_GAP_MINUTES = 60

# Physical gradient check (4.II.B.1)
# Max allowable physical slope between consecutive pings (in degrees)
MAX_SLOPE_DEG = 60.0  

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

# Bilateral spikes and biological scattering layer detection (4.II.B.3)
BILATERAL_WINDOW = 7      
BILATERAL_REL_THR = 0.08   
PLATEAU_MAX_PINGS = 15     # If depth jumps and stays flat for this many pings, flag as biological layer lock
BILATERAL_MIN_DEPTH = 2.0  

# SavGol Filter (4.II.B.4)
SG_WINDOW_SIZE = 51        # Must be an odd number
SG_POLY_ORDER = 2          # 2nd order polynomial to preserve peaks
SG_SIGMA_K = 4.0             # flag residuals more than this many robust-sigma above the median

DIRECT_TID_VALUES = set(range(10, 18)) # 4.II.B.5.b


def pick_paths_via_gui() -> dict:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
 
    print("Opening file manager - check terminal for prompt")
    csv_folder = filedialog.askdirectory(title="Select folder containing CSB CSV files")
    if not csv_folder: raise SystemExit("No CSV folder selected — exiting.")
 
    raster_path = filedialog.askopenfilename(
        title="Select reference bathymetry GeoTIFF",
        filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")]
    )
    if not raster_path: raise SystemExit("No reference raster selected — exiting.")
 
    use_tid = messagebox.askyesno("TID Filter", "Do you have a TID GeoTIFF for direct measurements?")
    tid_path = None
    if use_tid:
        tid_path = filedialog.askopenfilename(
            title="Select TID GeoTIFF", filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")]
        )
 
    out_dir = filedialog.askdirectory(title="Select output folder")
    if not out_dir: raise SystemExit("No output folder selected — exiting.")
 
    root.destroy()
    return dict(csv_folder=csv_folder, raster_path=raster_path, tid_path=tid_path, out_dir=out_dir)

def _parse_date_only(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.zfill(8)
    if s.str.match(r"^\d{8}$").all():
        # jumbled date with no separators, e.g. "25022024" -> 25-02-2024
        return pd.to_datetime(s, format="%d%m%Y", errors="coerce")
    return pd.to_datetime(series.astype(str).str.strip(), dayfirst=True, errors="coerce")


def _parse_time_of_day(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().all() and not s.str.contains(":").any():
        # second of day, e.g. 43200 -> 12:00:00
        return pd.to_timedelta(numeric, unit="s")
    if s.str.contains(r"(?i)\s*[ap]\.?m\.?\s*$", regex=True).any():
        # 12-hour clock with AM/PM suffix — pd.to_timedelta silently ignores
        # the AM/PM marker, so parse as a datetime and take the time-of-day part
        dt = pd.to_datetime(s, errors="coerce")
        return dt - dt.dt.normalize()
    return pd.to_timedelta(s, errors="coerce")


def _parse_single_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    required = {"lat", "lon", "depth"}
    if required - set(df.columns):
        raise ValueError(f"{os.path.basename(csv_path)}: missing columns")

    date_col = next((c for c in df.columns if c.strip().lower().startswith("date")), None)
    time_col = next((c for c in df.columns if c.strip().lower().startswith("time")), None)

    if date_col and time_col and date_col != time_col:
        df["time"] = _parse_date_only(df[date_col]) + _parse_time_of_day(df[time_col])
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], dayfirst=True, errors="coerce")
    
    if "unique_id" not in df.columns: df["unique_id"] = "unknown_vessel"
    if "platform_name" not in df.columns: df["platform_name"] = "unknown"

    df["depth"] = pd.to_numeric(df["depth"], errors="coerce").abs()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    df = df.dropna(subset=["time", "depth", "lat", "lon", "unique_id"])
    df = df[(df["depth"] > 0.5) & (df["depth"] < 11000)]
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
    return df.drop_duplicates(subset=["lon", "lat", "depth", "time", "unique_id"])

def load_csb(csv_folder: str) -> gpd.GeoDataFrame:
    csv_files = sorted(p for p in Path(csv_folder).iterdir() if p.suffix.lower() == ".csv")
    frames = [_parse_single_csv(str(p)) for p in csv_files]
    df = pd.concat(frames, ignore_index=True).sort_values(["unique_id", "time"]).reset_index(drop=True)
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

def assign_transit_ids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf["transit_id"] = ""
    gap = pd.Timedelta(minutes=TRANSIT_GAP_MINUTES)
    for uid, grp in gdf.groupby("unique_id"):
        segment = (grp["time"].diff() > gap).cumsum()
        for seg_num, seg_grp in grp.groupby(segment):
            gdf.loc[seg_grp.index, "transit_id"] = f"{uid}_T{int(seg_num):03d}"
    return gdf

# Outlier Detection Layers

def haversine(lon1, lat1, lon2, lat2):
    R = 6371000  # radius of Earth in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def flag_layer0_physical_gradient(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Layer 0: Checks for physically impossible slopes between consecutive pings."""
    new_flags = 0
    max_slope_rad = np.radians(MAX_SLOPE_DEG)

    for transit_id, grp in gdf.groupby("transit_id"):
        lons, lats = grp["lon"].values, grp["lat"].values
        depths = grp["depth"].values
        
        dist = haversine(lons[:-1], lats[:-1], lons[1:], lats[1:])
        ddepth = np.abs(depths[1:] - depths[:-1])
        
        # Calculate angle, avoiding div by zero
        safe_dist = np.where(dist == 0, 1e-6, dist)
        slopes = np.arctan(ddepth / safe_dist)
        
        # Shift mask to flag the anomalous point
        flag_mask = np.insert((slopes > max_slope_rad), 0, False)
        
        flagged_idx = grp.index[flag_mask]
        new_flags += len(flagged_idx)
        gdf.loc[flagged_idx, "outlier"] = True
        gdf.loc[flagged_idx, "outlier_reason"] = "physical_gradient_exceeded"

    print(f"    Layer 0 (Physical Gradient): {new_flags:,} points flagged")
    return gdf

def flag_layer1_asymmetric_ref(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Layer 1: Asymmetric reference check based on TID confidence."""
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

    print(f"    Layer 1 (Asymmetric Ref): {new_flags:,} points flagged")
    return gdf

def flag_layer2_biological_plateau(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Layer 2: Bilateral spike AND biological scattering layer plateau check."""
    new_flags = 0

    for transit_id, grp in gdf.groupby("transit_id"):
        depths = np.abs(grp["depth"].values.astype(float))
        n = len(depths)
        pre_flagged = grp["outlier"].values.copy()
        jump_flags = np.zeros(n, dtype=bool)

        for i in range(n):
            d = depths[i]
            if np.isnan(d) or d < BILATERAL_MIN_DEPTH or pre_flagged[i]: continue

            back_vals, fwd_vals = [], []
            j = i - 1
            while j >= 0 and len(back_vals) < BILATERAL_WINDOW:
                if not pre_flagged[j]: back_vals.append(depths[j])
                j -= 1
            j = i + 1
            while j < n and len(fwd_vals) < BILATERAL_WINDOW:
                if not pre_flagged[j]: fwd_vals.append(depths[j])
                j += 1

            if len(back_vals) < 2 or len(fwd_vals) < 2: continue

            back_med, fwd_med = np.median(back_vals), np.median(fwd_vals)
            local_ref = min(back_med, fwd_med)
            thr = max(5.0, local_ref * BILATERAL_REL_THR)

            dev_back, dev_fwd = abs(d - back_med), abs(d - fwd_med)

            # Standard spike
            if dev_back > thr and dev_fwd > thr:
                jump_flags[i] = True
            
            # Plateau check (biological layer lock)
            # If standard deviation within the local window is tight, but it heavily deviates 
            # from the extended edges, it might be tracking fish.
            elif len(back_vals) == BILATERAL_WINDOW and len(fwd_vals) == BILATERAL_WINDOW:
                local_std = np.std(fwd_vals[:PLATEAU_MAX_PINGS//2])
                if local_std < 5.0 and dev_back > (thr * 2):
                    jump_flags[i] = True

        idx = grp.index[jump_flags]
        gdf.loc[idx, "outlier"] = True
        gdf.loc[idx, "outlier_reason"] = "spike_or_plateau"
        new_flags += len(idx)

    print(f"    Layer 2 (Spike/Plateau): {new_flags:,} points flagged")
    return gdf

def flag_layer3_savgol_residuals(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Layer 3: Savitzky-Golay polynomial smoothing residual."""
    new_flags = 0

    for transit_id, grp in gdf.groupby("transit_id"):
        if len(grp) < SG_WINDOW_SIZE: continue

        depths = grp["depth"].values.astype(float)
        flagged_mask = gdf.loc[grp.index, "outlier"].values
        work = depths.copy()

        # Interpolate already flagged points so they don't drag the polynomial
        valid_idx = np.where(~flagged_mask)[0]
        if len(valid_idx) < 2: continue
        work[flagged_mask] = np.interp(np.where(flagged_mask)[0], valid_idx, work[valid_idx])

        # Apply Savitzky-Golay
        smoothed = savgol_filter(work, window_length=SG_WINDOW_SIZE, polyorder=SG_POLY_ORDER)
        residuals = np.abs(depths - smoothed)
        
        valid_res = residuals[~flagged_mask]
        if len(valid_res) == 0: continue

        # Robust (MAD-based) absolute threshold, so a clean transit can
        # legitimately flag zero points instead of always flagging a fixed
        # top-percentile of its own residual distribution.
        med = np.median(valid_res)
        mad = np.median(np.abs(valid_res - med)) * 1.4826
        threshold = np.inf if mad < 1e-6 else med + SG_SIGMA_K * mad
        new_outlier = (residuals > threshold) & (~flagged_mask)
        
        idx = grp.index[new_outlier]
        gdf.loc[idx, "outlier"] = True
        gdf.loc[idx, "outlier_reason"] = "savgol_residual"
        new_flags += int(new_outlier.sum())

    print(f"    Layer 3 (SavGol Residual): {new_flags:,} points flagged")
    return gdf


# Plotting + export

REASON_STYLE = {
    "physical_gradient_exceeded":   dict(color="red",     marker="x", s=15, zorder=6, label="Layer 0: Physics"),
    "hard_ref_direct":              dict(color="magenta", marker="o", s=10, zorder=5, label="Layer 1: Ref Direct"),
    "hard_ref_indirect_asymmetric": dict(color="purple",  marker="o", s=10, zorder=5, label="Layer 1: Ref Indirect"),
    "spike_or_plateau":             dict(color="orange",  marker="o", s=10, zorder=4, label="Layer 2: Spike/Plateau"),
    "savgol_residual":              dict(color="cyan",    marker="o", s=10, zorder=4, label="Layer 3: SavGol Res"),
}

def export_plots(gdf: gpd.GeoDataFrame, plots_dir: str) -> None:
    os.makedirs(plots_dir, exist_ok=True)
    transits = gdf["transit_id"].unique()
    print(f"  Exporting {len(transits)} transit plots → {plots_dir}")
    
    for tid in transits:
        grp = gdf[gdf["transit_id"] == tid].sort_values("time").copy().reset_index()
        fig, ax = plt.subplots(figsize=(14, 6))
        
        times = pd.to_datetime(grp["time"])
        valid = grp[~grp["outlier"]]

        has_tid = "tid_value" in grp.columns
        if has_tid:
            ref_direct = grp[grp["has_reference"] & grp["tid_value"].apply(lambda t: pd.notna(t) and int(t) in DIRECT_TID_VALUES)]
            ref_indirect = grp[grp["has_reference"] & ~grp["tid_value"].apply(lambda t: pd.notna(t) and int(t) in DIRECT_TID_VALUES)]
        else:
            ref_direct = grp.iloc[0:0]
            ref_indirect = grp[grp["has_reference"]]

        if not valid.empty: ax.scatter(times[valid.index], -valid["depth"], color="blue", s=4, zorder=3, label="Valid")
        if not ref_direct.empty: ax.scatter(times[ref_direct.index], -ref_direct["raster_val"], color="black", s=4, zorder=2, label="Reference (direct)")
        if not ref_indirect.empty: ax.scatter(times[ref_indirect.index], -ref_indirect["raster_val"], color="gray", s=4, zorder=2, label="Reference (indirect)")
        
        for reason, style in REASON_STYLE.items():
            sub = grp[grp["outlier_reason"] == reason]
            if not sub.empty: ax.scatter(times[sub.index], -sub["depth"], **style)
        
        ax.set_ylabel("Depth (m)")
        ax.set_title(tid)
        ax.legend(loc="lower left", fontsize=7)
        fig.autofmt_xdate(rotation=25, ha="right")
        
        vessel = "".join(c if c.isalnum() else "_" for c in str(grp["platform_name"].iloc[0]))
        t_start = times.iloc[0].strftime("%Y%m%dT%H%M%S")
        t_end = times.iloc[-1].strftime("%Y%m%dT%H%M%S")
        plt.savefig(os.path.join(plots_dir, f"{vessel}_{tid}_{t_start}_{t_end}.png"), dpi=140, facecolor="white")
        plt.close(fig)

def main():
    paths = pick_paths_via_gui()
    OUT_DIR = paths["out_dir"]
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1/4] Loading Data...")
    gdf = load_csb(paths["csv_folder"])
    
    print("[2/4] Sampling Rasters...")
    gdf = sample_raster(gdf, paths["raster_path"], "raster_val")
    gdf["has_reference"] = gdf["raster_val"].notna()
    if paths["tid_path"]: gdf = sample_raster(gdf, paths["tid_path"], "tid_value")
    
    print("[3/4] Assigning Transits...")
    gdf = assign_transit_ids(gdf)
    
    print("[4/4] Outlier Detection Pipeline...")
    gdf["outlier"] = False
    gdf["outlier_reason"] = ""
    gdf = flag_layer0_physical_gradient(gdf)
    gdf = flag_layer1_asymmetric_ref(gdf)
    gdf = flag_layer2_biological_plateau(gdf)
    gdf = flag_layer3_savgol_residuals(gdf)

    print("\nExporting...")
    export_plots(gdf, os.path.join(OUT_DIR, "outlier_plots"))
    
    # Export GPKG
    out = gdf.copy()
    out["depth"] = -out["depth"]
    out["raster_val"] = -out["raster_val"]
    out["time"] = out["time"].astype(str)
    out.to_file(os.path.join(OUT_DIR, "csb_processed.gpkg"), driver="GPKG")

    # Export CSVs (same attributes as GPKG, minus geometry)
    out_csv = out.drop(columns="geometry")
    out_csv.to_csv(os.path.join(OUT_DIR, "csb_processed.csv"), index=False)
    out_csv[~out_csv["outlier"]].to_csv(os.path.join(OUT_DIR, "csb_processed_no_outliers.csv"), index=False)

    print("Processing Complete.")

if __name__ == "__main__":
    main()