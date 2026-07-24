"""
Rewrite of Anthony Klemm's CSB processing script to be more suitable for ocean water

Inputs

  - CSB CSV file  (columns: unique_id, platform_name, time, lat, lon, depth)
  - Reference bathymetry GeoTIFF  (with negative depth values)
  - Optional TID GeoTIFF          (restrict to only direct-measurement cells)

Processing

  1. Load & sanitise CSV
  2. Sample reference raster at each point
  3. Derive per-vessel systematic offset from reference (optional TID filter)
  4. Apply offset  →  depth  (positive, metres below surface)
  5. Outlier detection  (three independent layers, all non-destructive):
       a. Hard reference deviation  — immediate discard of extreme disagreements
       b. Bilateral median spike    — catches sounder glitches along-track
       c. Smoothing residual        — catches subtler scatter within each transit
  6. Export GeoPackage  (all points; outlier column lets you filter in QGIS)
  7. Export per-transit outlier plots (time on x-axis, depth on y-axis)

Outputs

  csb_processed_points.gpkg:
      depth          CSB depth (negative)
      raster_val     reference raster value at point location
      vessel_offset  per-vessel correction applied (0 if no reference)
      has_reference  True where a reference exists
      outlier        True if flagged by any detection layer
      outlier_reason label
      transit_id     vessel + time-gap segmentation

  outlier_plots/
      {unique_id}_{transit_id}.png  — one plot per transit
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.ndimage import uniform_filter1d
from shapely.geometry import Point
import tkinter as tk
from tkinter import filedialog, messagebox

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Parameters

# Transit splitting: a gap > this many minutes between consecutive pings
# in the same vessel creates a new transit segment.
TRANSIT_GAP_MINUTES = 60

# Hard reference deviation thresholds (layer a):
# Direct measurement cells
HARD_DEV_FRACTION_DIRECT = 0.10   # 10% of depth
HARD_DEV_MIN_M_DIRECT    = 5.0    # minimum absolute deviation to flag
HARD_DEV_ABS_CAP_DIRECT  = 100.0  # always flag if deviation exceeds this

# Indirect measurement cells
HARD_DEV_FRACTION_INDIRECT = 0.30   # 30% of depth
HARD_DEV_MIN_M_INDIRECT    = 20.0   # minimum absolute deviation to flag
HARD_DEV_ABS_CAP_INDIRECT  = 400.0  # always flag if deviation exceeds this

# Bilateral median spike detector (layer b):
BILATERAL_WINDOW  = 7      # neighbours on each side
BILATERAL_REL_THR = 0.08   # 8 % relative deviation (both sides must exceed)
BILATERAL_MIN_DEPTH = 2.0  # skip points shallower than this

# Smoothing residual (layer c):
SMOOTH_FILTER_SIZE = 50    # uniform filter window (pings)
SMOOTH_PERCENTILE  = 98    # flag residuals above this percentile

# TID values (inclusive-exclusive)
DIRECT_TID_VALUES = set(range(10, 18))

def pick_paths_via_gui() -> dict:
    # use native file picker to select input/output paths
    root = tk.Tk()
    root.withdraw()   # hide the empty root window
    root.attributes("-topmost", True)
 
    print("Opening file manager - check terminal for prompt")
 
    # CSV folder
    print("Select folder containing CSB CSV files")
    csv_folder = filedialog.askdirectory(title="Select folder containing CSB CSV files")
    if not csv_folder:
        raise SystemExit("No CSV folder selected — exiting.")
 
    # Reference raster
    print("Select reference bathymetry GeoTIFF")
    raster_path = filedialog.askopenfilename(
        title="Select reference bathymetry GeoTIFF",
        filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")]
    )
    if not raster_path:
        raise SystemExit("No reference raster selected — exiting.")
 
    # TID raster (optional)
    use_tid = messagebox.askyesno(
        "TID Filter",
        "Do you have a TID GeoTIFF for more accurate outlier detection on points with direct measurements?"
    )
    tid_path = None
    if use_tid:
        print("Select TID GeoTIFF")
        tid_path = filedialog.askopenfilename(
            title="Select TID GeoTIFF",
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if not tid_path:
            print("    [info] No TID file selected — TID filter disabled.")
            tid_path = None
 
    # Output folder
    print("Select output folder (will be created if needed)")
    out_dir = filedialog.askdirectory(title="Select output folder")
    if not out_dir:
        raise SystemExit("No output folder selected — exiting.")
 
    root.destroy()
    return dict(csv_folder=csv_folder, raster_path=raster_path,
                tid_path=tid_path, out_dir=out_dir)

# Load CSV + premliminary sanity checks

def _parse_single_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"lat", "lon", "depth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(csv_path)}: missing columns {missing}")

    # time parsing, prefer them split up into date and time columns but accept a single time column if present
    date_col = next((c for c in df.columns if c.strip().lower().startswith("date")), None)
    time_col = next((c for c in df.columns if c.strip().lower().startswith("time")), None)

    if date_col and time_col and date_col != time_col:
        combined = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
        df["time"] = pd.to_datetime(combined, dayfirst=True, errors="coerce")
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], dayfirst=True, errors="coerce")
    else:
        raise ValueError(f"{os.path.basename(csv_path)}: no time or date column found")

    # optional cols
    if "unique_id" not in df.columns:
        df["unique_id"] = "unknown_vessel"
    if "platform_name" not in df.columns:
        df["platform_name"] = "unknown"

    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
    df["lat"]   = pd.to_numeric(df["lat"],   errors="coerce")
    df["lon"]   = pd.to_numeric(df["lon"],   errors="coerce")
    df['depth'] = df['depth'].abs()  # ensure positive depth values

    before = len(df)
    df = df.dropna(subset=["time", "depth", "lat", "lon", "unique_id"])
    df = df[(df["depth"] > 0.5) & (df["depth"] < 11000)]
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
    df = df.drop_duplicates(subset=["lon", "lat", "depth", "time", "unique_id"])
    dropped = before - len(df)
    print(f"    {os.path.basename(csv_path)}: {len(df):,} rows ({dropped:,} dropped)")
    return df


def load_csb(csv_folder: str) -> gpd.GeoDataFrame:
    csv_files = sorted(
        p for p in Path(csv_folder).iterdir()
        if p.suffix.lower() == ".csv"
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {csv_folder}")

    print(f"[1/5] Loading {len(csv_files)} CSV file(s) from: {csv_folder}")

    frames = []
    for csv_path in csv_files:
        try:
            frames.append(_parse_single_csv(str(csv_path)))
        except Exception as e:
            print(f"    [warn] Skipping {csv_path.name}: {e}")

    if not frames:
        raise RuntimeError("All CSV files failed to load")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["unique_id", "time"]).reset_index(drop=True)
    print(f"    Total: {len(df):,} rows across {len(frames)} file(s)")

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )


# Reference Raster

def sample_raster(gdf: gpd.GeoDataFrame,
                  raster_path: str,
                  col_name: str) -> gpd.GeoDataFrame:
    coords = [(geom.x, geom.y) for geom in gdf.geometry]
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        sampled = [v[0] for v in src.sample(coords)]

    values = np.array(sampled, dtype=float)

    # Mark nodata as NaN
    if nodata is not None:
        values[np.isclose(values, nodata)] = np.nan
    values[values == 0] = np.nan   # zero depth is also invalid

    values = np.abs(values)

    gdf = gdf.copy()
    gdf[col_name] = values
    return gdf


def sample_tid(gdf: gpd.GeoDataFrame, tid_path: str) -> gpd.GeoDataFrame:
    """Sample TID grid; store raw integer TID values."""
    coords = [(geom.x, geom.y) for geom in gdf.geometry]
    with rasterio.open(tid_path) as src:
        nodata = src.nodata
        sampled = [v[0] for v in src.sample(coords)]

    values = np.array(sampled, dtype=float)
    if nodata is not None:
        values[np.isclose(values, nodata)] = np.nan

    gdf = gdf.copy()
    gdf["tid_value"] = values
    return gdf



def assign_transit_ids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["transit_id"] = ""
    gap = pd.Timedelta(minutes=TRANSIT_GAP_MINUTES)

    for uid, grp in gdf.groupby("unique_id"):
        grp = grp.sort_values("time")
        time_diff = grp["time"].diff()
        segment = (time_diff > gap).cumsum()
        for seg_num, seg_grp in grp.groupby(segment):
            tid = f"{uid}_T{int(seg_num):03d}"
            gdf.loc[seg_grp.index, "transit_id"] = tid

    return gdf


# Outlier Detection

def flag_hard_reference_deviations(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Layer (a) — Hard reference deviation, split by TID confidence.
    Direct cells (TID 10-17): tight threshold.
    Indirect cells (TID 18+ or no TID): loose threshold.
    """
    gdf = gdf.copy()
    mask_ref = gdf["has_reference"]
    has_tid  = "tid_value" in gdf.columns

    if has_tid:
        direct_mask   = mask_ref & gdf["tid_value"].apply(
            lambda t: pd.notna(t) and int(t) in DIRECT_TID_VALUES
        )
        indirect_mask = mask_ref & ~direct_mask
    else:
        direct_mask   = pd.Series(False, index=gdf.index)
        indirect_mask = mask_ref

    flagged_total = 0
    for mask, frac, min_m, cap, label in [
        (direct_mask,   HARD_DEV_FRACTION_DIRECT,   HARD_DEV_MIN_M_DIRECT,
                        HARD_DEV_ABS_CAP_DIRECT,   "direct"),
        (indirect_mask, HARD_DEV_FRACTION_INDIRECT, HARD_DEV_MIN_M_INDIRECT,
                        HARD_DEV_ABS_CAP_INDIRECT, "indirect"),
    ]:
        if not mask.any():
            continue
        dev = np.abs(gdf.loc[mask, "depth"] - gdf.loc[mask, "raster_val"])
        thr = np.maximum(min_m, gdf.loc[mask, "depth"] * frac)
        flag = (dev > thr) | (dev > cap)
        flagged_idx = gdf.loc[mask][flag].index
        gdf.loc[flagged_idx, "outlier"] = True
        gdf.loc[flagged_idx, "outlier_reason"] = f"hard_ref_deviation_{label}"
        count = flag.sum()
        flagged_total += count
        print(f"    Layer (a) hard ref deviation [{label:8s}]: {count:,} points flagged")

    print(f"    Layer (a) total: {flagged_total:,} points flagged")
    return gdf


def flag_bilateral_median_spikes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Layer 2: Bilateral median spike detector
    gdf = gdf.copy()
    new_flags = 0

    for transit_id, grp in gdf.groupby("transit_id"):
        depths_raw = grp["depth"].values.astype(float)
        depths = np.abs(depths_raw)
        n = len(depths)
        pre_flagged = grp["outlier"].values.copy()
        jump_flags  = np.zeros(n, dtype=bool)

        for i in range(n):
            d = depths[i]
            if np.isnan(d) or d < BILATERAL_MIN_DEPTH:
                continue

            back_vals, fwd_vals = [], []
            j = i - 1
            while j >= 0 and len(back_vals) < BILATERAL_WINDOW:
                if not np.isnan(depths[j]) and not pre_flagged[j]:
                    back_vals.append(depths[j])
                j -= 1
            j = i + 1
            while j < n and len(fwd_vals) < BILATERAL_WINDOW:
                if not np.isnan(depths[j]) and not pre_flagged[j]:
                    fwd_vals.append(depths[j])
                j += 1

            if len(back_vals) < 2 or len(fwd_vals) < 2:
                continue

            back_med = np.median(back_vals)
            fwd_med  = np.median(fwd_vals)
            # Threshold based on neighbourhood depth, not the candidate point
            local_ref = min(back_med, fwd_med)
            thr = max(5.0, local_ref * 0.05)

            dev_back = abs(d - back_med)
            dev_fwd  = abs(d - fwd_med)
            rel_back = dev_back / max(back_med, 1e-6)
            rel_fwd  = dev_fwd  / max(fwd_med,  1e-6)

            if (dev_back > thr and dev_fwd > thr and
                    rel_back > BILATERAL_REL_THR and rel_fwd > BILATERAL_REL_THR):
                jump_flags[i] = True

        flagged_idx = grp.index[jump_flags]
        already = gdf.loc[flagged_idx, "outlier"]
        new_idx = flagged_idx[~already]
        gdf.loc[new_idx, "outlier"] = True
        gdf.loc[new_idx, "outlier_reason"] = "bilateral_spike"
        new_flags += len(new_idx)

    print(f"    Layer (b) bilateral median spike: {new_flags:,} new points flagged")
    return gdf


def flag_smoothing_residuals(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Layer 3: Smoothing residual detector
    gdf = gdf.copy()
    new_flags = 0

    for transit_id, grp in gdf.groupby("transit_id"):
        if len(grp) < SMOOTH_FILTER_SIZE // 2:
            continue   # too short to smooth meaningfully

        depths = grp["depth"].values.astype(float)

        for _pass in range(2):
            flagged_mask = gdf.loc[grp.index, "outlier"].values
            work = depths.copy()
            # Replace already-flagged points with local median so the smoother
            # isn't pulled toward bad values
            for i in np.where(flagged_mask)[0]:
                neighbours = np.concatenate([depths[max(0, i-5) : i], depths[i+1 : min(len(depths), i+6)]])
                valid = [v for v in neighbours if not np.isnan(v)]
                work[i] = np.median(valid) if valid else np.nanmedian(depths)

            work = np.where(np.isnan(work), np.nanmedian(depths), work)
            smoothed  = uniform_filter1d(work, size=SMOOTH_FILTER_SIZE)
            residuals = np.abs(depths - smoothed)
            # Only compute threshold from non-flagged points
            valid_res = residuals[~flagged_mask]
            if len(valid_res) == 0:
                continue
            threshold = np.percentile(valid_res, SMOOTH_PERCENTILE)
            new_outlier = (residuals > threshold) & (~flagged_mask)
            flagged_idx = grp.index[new_outlier]
            gdf.loc[flagged_idx, "outlier"] = True
            gdf.loc[flagged_idx, "outlier_reason"] = gdf.loc[
                flagged_idx, "outlier_reason"
            ].where(gdf.loc[flagged_idx, "outlier_reason"] != "", "smoothing_residual")
            gdf.loc[flagged_idx[
                gdf.loc[flagged_idx, "outlier_reason"] == ""
            ], "outlier_reason"] = "smoothing_residual"
            new_flags += int(new_outlier.sum())

    print(f"    Layer (c) smoothing residual: {new_flags:,} new points flagged")
    return gdf


def run_outlier_detection(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["outlier"]        = False
    gdf["outlier_reason"] = ""

    print("  Running outlier detection layers...")
    gdf = flag_hard_reference_deviations(gdf)
    gdf = flag_bilateral_median_spikes(gdf)
    gdf = flag_smoothing_residuals(gdf)

    total = gdf["outlier"].sum()
    pct   = 100 * total / max(len(gdf), 1)
    print(f"  Total flagged: {total:,} / {len(gdf):,} ({pct:.1f}%)")
    return gdf


# Create Outlier Plots

REASON_STYLE = {
    "hard_ref_deviation_direct":   dict(color="magenta",    marker="o", s=10, zorder=5, label="Hard ref deviation (direct)"),
    "hard_ref_deviation_indirect": dict(color="magenta",    marker="o", s=10, zorder=5, label="Hard ref deviation (indirect)"),
    "bilateral_spike":             dict(color="orange", marker="o", s=10, zorder=5, label="Bilateral spike"),
    "smoothing_residual":          dict(color="orange", marker="o", s=10, zorder=4, label="Smoothing residual"),
    "valid":                       dict(color="blue",  marker="o", s=4,  zorder=3, label="Valid"),
}
 
def plot_transit(grp: pd.DataFrame, transit_id: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.tick_params(colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
 
    times = pd.to_datetime(grp["time"])
    depth_display  = -grp["depth"]
    raster_display = -grp["raster_val"]
 
    # Valid points
    valid = grp[~grp["outlier"]]
    if not valid.empty:
        ax.scatter(times[valid.index], depth_display[valid.index],
                   **{**REASON_STYLE["valid"], "label": "Valid"})
 
    # Reference raster — black dots
    ref = grp[grp["has_reference"]]
    if not ref.empty:
        ax.scatter(times[ref.index], raster_display[ref.index],
                   color="black", s=4, marker="o", zorder=2,
                   label="Reference raster")
 
    # Flagged points by reason
    for reason, style in REASON_STYLE.items():
        if reason == "valid":
            continue
        sub = grp[grp["outlier"] & (grp["outlier_reason"] == reason)]
        if not sub.empty:
            ax.scatter(times[sub.index], depth_display[sub.index], **style)
 
    ax.set_ylabel("Depth (m)", fontsize=9)
    ax.set_xlabel("Time", fontsize=9)
    ax.set_title(transit_id, fontsize=10, pad=6)
    ax.legend(loc="lower left", fontsize=7)
 
    fig.autofmt_xdate(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)

def export_plots(gdf: gpd.GeoDataFrame, plots_dir: str) -> None:
    os.makedirs(plots_dir, exist_ok=True)
    transits = gdf["transit_id"].unique()
    print(f"  Exporting {len(transits)} transit plots → {plots_dir}")
    for tid in transits:
        grp = gdf[gdf["transit_id"] == tid].sort_values("time").copy().reset_index()
        try:
            # better filenames for readability
            vessel   = grp["platform_name"].iloc[0] if "platform_name" in grp.columns else "unknown"
            t_start  = pd.to_datetime(grp["time"].iloc[0]).strftime("%Y%m%d_%H%M%S")
            t_end    = pd.to_datetime(grp["time"].iloc[-1]).strftime("%Y%m%d_%H%M%S")
            safe_vessel = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(vessel))
            filename = f"{safe_vessel}_{t_start}_to_{t_end}.png"
            plot_transit(grp, tid, os.path.join(plots_dir, filename))
        except Exception as e:
            print(f"    [warn] could not plot {tid}: {e}")
 
 
EXPORT_COLS = [
    "unique_id", "platform_name", "time",
    "lat", "lon",
    "depth", "raster_val", "ref_difference",
    "has_reference",
    "outlier", "outlier_reason",
    "transit_id", "geometry"
]
 
 
def export_gpkg(gdf: gpd.GeoDataFrame, out_path: str) -> None:
    out = gdf[[c for c in EXPORT_COLS if c in gdf.columns]].copy()

    # ref_difference: positive = CSB deeper than reference, negative = shallower
    out["ref_difference"] = out["depth"] - out["raster_val"]
    out.loc[~out["has_reference"], "ref_difference"] = np.nan

    # make sure they are negative
    out["depth"] = -out["depth"]
    out["raster_val"] = -out["raster_val"]

    out["time"] = out["time"].astype(str)
    out.to_file(out_path, driver="GPKG")

    n_valid   = (~out["outlier"]).sum()
    n_flagged = out["outlier"].sum()
    print(f"  GeoPackage saved: {out_path}")
    print(f"  {n_valid:,} valid  |  {n_flagged:,} flagged as outlier")
    for reason, count in out[out["outlier"]]["outlier_reason"].value_counts().items():
        print(f"    {reason}: {count:,}")

    ref_diff = out["ref_difference"].dropna()
    if len(ref_diff) > 0:
        print(f"  CSB vs reference ({len(ref_diff):,} points with reference):")
        print(f"    Mean difference (bias): {ref_diff.mean():.3f} m")
        print(f"    Std deviation         : {ref_diff.std():.3f} m")
        print(f"    Min / Max             : {ref_diff.min():.3f} / {ref_diff.max():.3f} m")

# Main method

def main():
    # Pick files/folders via native OS dialogs
    paths = pick_paths_via_gui()
    CSV_FOLDER  = paths["csv_folder"]
    RASTER_PATH = paths["raster_path"]
    TID_PATH    = paths["tid_path"]
    OUT_DIR     = paths["out_dir"]
 
    plots_dir = os.path.join(OUT_DIR, "outlier_plots")
    gpkg_path = os.path.join(OUT_DIR, "csb_processed_points.gpkg")
    os.makedirs(OUT_DIR, exist_ok=True)

    use_tid = TID_PATH is not None

    # Load
    gdf = load_csb(CSV_FOLDER)

    # Sample reference raster
    print(f"[2/4] Sampling reference raster: {RASTER_PATH}")
    gdf = sample_raster(gdf, RASTER_PATH, col_name="raster_val")
    gdf["has_reference"] = gdf["raster_val"].notna()
    n_ref = gdf["has_reference"].sum()
    print(f"    {n_ref:,}/{len(gdf):,} points have a reference raster value")

    if use_tid:
        print(f"    Sampling TID grid: {TID_PATH}")
        gdf = sample_tid(gdf, TID_PATH)
        n_direct = gdf["tid_value"].apply(
            lambda t: pd.notna(t) and int(t) in DIRECT_TID_VALUES
        ).sum()
        print(f"    {n_direct:,} points on direct-measurement cells (TID 10-17)")
        print(f"    {n_ref - n_direct:,} points on indirect/predicted cells (TID 18+)")

    # Prepare depths and segment transits
    print(f"[3/4] Preparing depths and assigning transit IDs")
    gdf = assign_transit_ids(gdf)
    print(f"    {gdf['transit_id'].nunique():,} transit segments identified")

    # Detect outliers
    print(f"[4/4] Running outlier detection")
    gdf = run_outlier_detection(gdf)
 
 
    # Export
    print(f"\nExporting results → {OUT_DIR}")
    export_gpkg(gdf, gpkg_path)
    export_plots(gdf, plots_dir)
 
    print("\nFinished processing")
 
 
if __name__ == "__main__":
    main()