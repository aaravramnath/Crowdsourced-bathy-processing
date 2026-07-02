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
  4. Apply offset  →  depth_corr  (positive, metres below surface)
  5. Outlier detection  (three independent layers, all non-destructive):
       a. Hard reference deviation  — immediate discard of extreme disagreements
       b. Bilateral median spike    — catches sounder glitches along-track
       c. Smoothing residual        — catches subtler scatter within each transit
  6. Export GeoPackage  (all points; outlier column lets you filter in QGIS)
  7. Export per-transit outlier plots (time on x-axis, depth on y-axis)

Outputs

  csb_processed_points.gpkg:
      depth_corr     corrected depth (positive, metres)
      depth_raw      original CSV depth
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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Parameters

# Transit splitting: a gap > this many minutes between consecutive pings
# in the same vessel creates a new transit segment.
TRANSIT_GAP_MINUTES = 60

# Vessel offset limits: offsets outside this range are considered unreliable
OFFSET_MIN_M = -11.0   # don't correct by more than 11 m shallow
OFFSET_MAX_M =  3.0    # don't correct by more than 3 m deep
OFFSET_MIN_STD_M = 7.0 # discard offset if std dev of diffs exceeds this

# Hard reference deviation threshold (layer a):
# Lower fraction to make more sensitive

HARD_DEV_FRACTION = 0.20   # 20 % of depth
HARD_DEV_MIN_M    = 10.0   # minimum absolute deviation to flag (metres)
# Additionally, points where the absolute deviation exceeds this value are
# immediately flagged regardless of fraction (catches near-zero dropouts
# in deep water where 20% might be a very large number).
HARD_DEV_ABS_CAP  = 200.0  # metres — always flag if deviation exceeds this

# Bilateral median spike detector (layer b):
BILATERAL_WINDOW  = 7      # neighbours on each side
BILATERAL_REL_THR = 0.08   # 8 % relative deviation (both sides must exceed)
BILATERAL_MIN_DEPTH = 2.0  # skip points shallower than this (abs metres)

# Smoothing residual (layer c):
SMOOTH_FILTER_SIZE = 50    # uniform filter window (pings)
SMOOTH_PERCENTILE  = 98    # flag residuals above this percentile

# TID values (inclusive-exclusive)
DIRECT_TID_VALUES = set(range(10, 18))


def load_csb(csv_path: str) -> gpd.GeoDataFrame:
    print(f"[1/5] Loading CSB CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    required = {"unique_id", "platform_name", "time", "lat", "lon", "depth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
    df["lat"]   = pd.to_numeric(df["lat"],   errors="coerce")
    df["lon"]   = pd.to_numeric(df["lon"],   errors="coerce")
    df["time"]  = pd.to_datetime(df["time"],  errors="coerce")

    before = len(df)
    df = df.dropna(subset=["time", "depth", "lat", "lon", "unique_id"])
    df = df[(df["depth"] > 0.5) & (df["depth"] < 11000)]
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
    df = df.drop_duplicates(subset=["lon", "lat", "depth", "time", "unique_id"])
    df = df.sort_values(["unique_id", "time"]).reset_index(drop=True)

    print(f"    {len(df):,} rows loaded ({before - len(df):,} dropped during sanity checks)")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )
    return gdf


# Reference Raster

def sample_raster(gdf: gpd.GeoDataFrame,
                  raster_path: str,
                  col_name: str) -> gpd.GeoDataFrame:
    """
    Sample a single-band raster at every point in gdf.
    Returns gdf with a new column col_name (NaN where outside raster extent).
    The raster is assumed to store depths as NEGATIVE values (GEBCO convention).
    We store the absolute value so downstream code always works in positive metres.
    """
    coords = [(geom.x, geom.y) for geom in gdf.geometry]
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        sampled = [v[0] for v in src.sample(coords)]

    values = np.array(sampled, dtype=float)

    # Mark nodata as NaN
    if nodata is not None:
        values[np.isclose(values, nodata)] = np.nan
    values[values == 0] = np.nan   # zero depth is also invalid

    # Convert from GEBCO negative convention → positive metres
    # (if raster already stores positives this is a no-op since we abs())
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


# Vessel Offset Calculations

def derive_vessel_offsets(gdf: gpd.GeoDataFrame,
                          use_tid_filter: bool) -> dict:
    ref = gdf.dropna(subset=["raster_val"]).copy()

    if use_tid_filter:
        before = len(ref)
        ref = ref[ref["tid_value"].apply(
            lambda t: int(t) in DIRECT_TID_VALUES if pd.notna(t) else False
        )]
        print(f"    TID filter: {len(ref):,}/{before:,} points on direct-measurement cells")

    ref["diff"] = ref["depth"] - ref["raster_val"]
    ref = ref[(ref["diff"] >= OFFSET_MIN_M) & (ref["diff"] <= OFFSET_MAX_M)]

    offsets = {}
    stats_rows = []

    for uid, grp in ref.groupby("unique_id"):
        mean_diff = grp["diff"].mean()
        std_diff  = grp["diff"].std()
        count     = len(grp)

        if std_diff > OFFSET_MIN_STD_M or count < 5:
            offset = 0.0
            note = f"rejected (std={std_diff:.2f}, n={count})"
        else:
            offset = mean_diff
            note = f"accepted (mean={mean_diff:.3f}m, std={std_diff:.2f}, n={count})"

        offsets[uid] = offset
        stats_rows.append({
            "unique_id": uid,
            "platform_name": grp["platform_name"].iloc[0],
            "offset_m": offset,
            "std_m": std_diff,
            "n_points": count,
            "note": note
        })
        print(f"    {uid[:40]:<40}  {note}")

    return offsets, pd.DataFrame(stats_rows)


# Apply Offsets

def apply_offsets(gdf: gpd.GeoDataFrame, offsets: dict) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["vessel_offset"] = gdf["unique_id"].map(offsets).fillna(0.0)
    gdf["depth_corr"] = gdf["depth"] - gdf["vessel_offset"]
    gdf["depth_raw"]  = gdf["depth"]
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
    # Layer 1: Hard reference deviation 
    gdf = gdf.copy()
    mask_ref = gdf["has_reference"]

    dev = np.abs(gdf.loc[mask_ref, "depth_corr"] - gdf.loc[mask_ref, "raster_val"])
    proportional_thr = np.maximum(
        HARD_DEV_MIN_M,
        gdf.loc[mask_ref, "depth_corr"] * HARD_DEV_FRACTION
    )

    hard_flag = (dev > proportional_thr) | (dev > HARD_DEV_ABS_CAP)
    flagged_idx = gdf.loc[mask_ref][hard_flag].index

    count = hard_flag.sum()
    gdf.loc[flagged_idx, "outlier"] = True
    gdf.loc[flagged_idx, "outlier_reason"] = "hard_reference_deviation"
    print(f"    Layer (a) hard reference deviation: {count:,} points flagged")
    return gdf


def flag_bilateral_median_spikes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Layer 2: Bilateral median spike detector
    gdf = gdf.copy()
    new_flags = 0

    for transit_id, grp in gdf.groupby("transit_id"):
        depths_raw = grp["depth_corr"].values.astype(float)
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

        depths = grp["depth_corr"].values.astype(float)

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
    "hard_reference_deviation": dict(color="#e63946", marker="s", s=20, zorder=5, label="Hard ref deviation"),
    "bilateral_spike":          dict(color="#f4a261", marker="^", s=20, zorder=5, label="Bilateral spike"),
    "smoothing_residual":       dict(color="#e9c46a", marker="o", s=12, zorder=4, label="Smoothing residual"),
    "valid":                    dict(color="#457b9d", marker="o", s=4,  zorder=3, label="Valid"),
}


def plot_transit(grp: pd.DataFrame, transit_id: str, out_path: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    times = pd.to_datetime(grp["time"])

    # top panel
    ax = axes[0]

    # Valid points
    valid = grp[~grp["outlier"]]
    if not valid.empty:
        ax.scatter(times[valid.index], valid["depth_corr"],
                   **{**REASON_STYLE["valid"], "label": "Valid"})

    # Reference raster line where available
    ref = grp[grp["has_reference"] & ~grp["outlier"]]
    if not ref.empty:
        ref_sorted = ref.sort_values("time")
        ax.plot(pd.to_datetime(ref_sorted["time"]), ref_sorted["raster_val"],
                color="#2ec4b6", linewidth=0.8, alpha=0.6, zorder=2,
                label="Reference raster")

    # Flagged points by reason
    for reason, style in REASON_STYLE.items():
        if reason == "valid":
            continue
        sub = grp[grp["outlier"] & (grp["outlier_reason"] == reason)]
        if not sub.empty:
            ax.scatter(times[sub.index], sub["depth_corr"], **style)

    ax.set_ylabel("Depth (m, positive down)", fontsize=9)
    ax.set_title(f"{transit_id}", color="#c9d1d9", fontsize=10, pad=6)
    ax.invert_yaxis()
    ax.legend(loc="lower left", fontsize=7, framealpha=0.3,
              labelcolor="#c9d1d9", facecolor="#161b22")

    # bottom panel: outlier reason strip
    ax2 = axes[1]
    reason_map = {"": 0, "hard_reference_deviation": 3,
                  "bilateral_spike": 2, "smoothing_residual": 1}
    colours_map = {"": "#457b9d", "hard_reference_deviation": "#e63946",
                   "bilateral_spike": "#f4a261", "smoothing_residual": "#e9c46a"}
    y_vals   = grp["outlier_reason"].map(reason_map).fillna(0)
    c_vals   = grp["outlier_reason"].map(colours_map).fillna("#457b9d")
    ax2.scatter(times, y_vals, c=c_vals, s=6, zorder=3)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(["Valid", "Smooth", "Spike", "Hard ref"],
                        fontsize=7, color="#c9d1d9")
    ax2.set_ylabel("Flag type", fontsize=8)
    ax2.set_xlabel("Time (UTC)", fontsize=9)

    fig.autofmt_xdate(rotation=25, ha="right")
    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def export_plots(gdf: gpd.GeoDataFrame, plots_dir: str) -> None:
    os.makedirs(plots_dir, exist_ok=True)
    transits = gdf["transit_id"].unique()
    print(f"  Exporting {len(transits)} transit plots → {plots_dir}")
    for tid in transits:
        grp = gdf[gdf["transit_id"] == tid].sort_values("time").copy()
        grp = grp.reset_index()   # keep original index in column
        safe_tid = tid.replace("/", "_").replace(":", "-")
        out_path = os.path.join(plots_dir, f"{safe_tid}.png")
        try:
            plot_transit(grp, tid, out_path)
        except Exception as e:
            print(f"    [warn] could not plot {tid}: {e}")


# Export Products

EXPORT_COLS = [
    "unique_id", "platform_name", "time",
    "lat", "lon",
    "depth_raw", "depth_corr",
    "raster_val",
    "vessel_offset",
    "has_reference",
    "outlier", "outlier_reason",
    "transit_id",
    "geometry"
]


def export_gpkg(gdf: gpd.GeoDataFrame, out_path: str) -> None:
    out = gdf[[c for c in EXPORT_COLS if c in gdf.columns]].copy()
    out["time"] = out["time"].astype(str)
    out.to_file(out_path, driver="GPKG")
    n_valid   = (~out["outlier"]).sum()
    n_flagged = out["outlier"].sum()
    print(f"  GeoPackage saved: {out_path}")
    print(f"  {n_valid:,} valid  |  {n_flagged:,} flagged as outlier")
    by_reason = out[out["outlier"]]["outlier_reason"].value_counts()
    for reason, count in by_reason.items():
        print(f"    {reason}: {count:,}")


# Main method

def main():
    # ── CONFIG — edit these paths before hitting Run ──────────────────────
    CSV_PATH    = r"data.csv"
    RASTER_PATH = r"gebco_bathymetry.tif"
    TID_PATH    = r"gebco_tid.tif"   # set to None to disable TID filter
    OUT_DIR     = r"output"
    # ─────────────────────────────────────────────────────────────────────

    out_dir    = OUT_DIR
    plots_dir  = os.path.join(out_dir, "outlier_plots")
    gpkg_path  = os.path.join(out_dir, "csb_processed_points.gpkg")
    offset_csv = os.path.join(out_dir, "vessel_offsets.csv")
    os.makedirs(out_dir, exist_ok=True)

    use_tid = TID_PATH is not None

    # Validate inputs early so errors are obvious
    for label, path in [("CSV", CSV_PATH), ("Raster", RASTER_PATH)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} file not found: {path}")
    if use_tid and not os.path.exists(TID_PATH):
        raise FileNotFoundError(f"TID file not found: {TID_PATH}")

    # Load
    gdf = load_csb(CSV_PATH)

    # Get reference raster values at each point
    print(f"[2/5] Sampling reference raster: {RASTER_PATH}")
    gdf = sample_raster(gdf, RASTER_PATH, col_name="raster_val")
    gdf["has_reference"] = gdf["raster_val"].notna()
    n_ref = gdf["has_reference"].sum()
    print(f"    {n_ref:,}/{len(gdf):,} points have a reference raster value")

    if use_tid:
        print(f"    Sampling TID grid: {TID_PATH}")
        gdf = sample_tid(gdf, TID_PATH)

    # Derive offsets
    print(f"[3/5] Deriving per-vessel offsets (TID filter: {'ON' if use_tid else 'OFF'})")
    offsets, offset_stats = derive_vessel_offsets(gdf, use_tid_filter=use_tid)
    offset_stats.to_csv(offset_csv, index=False)
    print(f"    Offset stats saved: {offset_csv}")

    # Apply offsets
    print(f"[4/5] Applying offsets and assigning transit IDs")
    gdf = apply_offsets(gdf, offsets)
    gdf = assign_transit_ids(gdf)
    n_transits = gdf["transit_id"].nunique()
    print(f"    {n_transits:,} transit segments identified")

    # Detect Outliers
    print(f"[5/5] Running outlier detection")
    gdf = run_outlier_detection(gdf)

    # Export
    print(f"\nExporting results → {out_dir}")
    export_gpkg(gdf, gpkg_path)
    export_plots(gdf, plots_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()