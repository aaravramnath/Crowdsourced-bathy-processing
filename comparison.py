# leveraging older code from other dataset to compare csb with predicted gebco bathymetry

import geopandas as gpd
import rasterio
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats


csb_gpkg = 'output/csb_processed_points.gpkg' 
gebco_bathy_file = 'gebco_bathymetry.tif'
gebco_tid_file = 'gebco_tid.tif'


depth_col = 'depth_corr' 


print("Loading CSB GeoPackage...")
csb_gdf = gpd.read_file(csb_gpkg)

with rasterio.open(gebco_tid_file) as src_tid:
    raster_crs = src_tid.crs
    
    if csb_gdf.crs != raster_crs:
        print(f"Reprojecting CSB data from {csb_gdf.crs} to {raster_crs}...")
        csb_gdf = csb_gdf.to_crs(raster_crs)

    # Extract X/Y coordinates from the point geometries
    coords = [(geom.x, geom.y) for geom in csb_gdf.geometry]
    
    # Sample the TID raster
    print("Sampling GEBCO TID grid...")
    csb_gdf['tid'] = [val[0] for val in src_tid.sample(coords)]


print("Filtering for predicted space (TID > 17)...")
predicted_gdf = csb_gdf[csb_gdf['tid'] > 17].copy()

if len(predicted_gdf) == 0:
    print("No CSB points intersect with GEBCO predicted space (TID > 17). Exiting.")
    exit()

print(f"Points falling strictly on predicted space: {len(predicted_gdf)}")

filtered_coords = [(geom.x, geom.y) for geom in predicted_gdf.geometry]

print("Sampling GEBCO bathymetry for the filtered points...")
with rasterio.open(gebco_bathy_file) as src_bathy:
    predicted_gdf['gebco_depth'] = [val[0] for val in src_bathy.sample(filtered_coords)]

predicted_gdf = predicted_gdf.dropna(subset=['gebco_depth', depth_col])

print("Filtering out GEBCO NaN values")
predicted_gdf = predicted_gdf[predicted_gdf['gebco_depth'] != -32767]
predicted_gdf = predicted_gdf[predicted_gdf['gebco_depth'] != -32768]

print(f"Valid points remaining for math: {len(predicted_gdf)}")


# If GEBCO is negative and CSB is positive, uncomment the next line:
predicted_gdf['gebco_depth'] = predicted_gdf['gebco_depth'] * -1

predicted_gdf['difference'] = predicted_gdf[depth_col] - predicted_gdf['gebco_depth']

print("Applying post-processing filter (removing differences > 500m or < -500m)...")
points_before = len(predicted_gdf)


predicted_gdf = predicted_gdf[(predicted_gdf['difference'] >= -500) & (predicted_gdf['difference'] <= 500)]

points_after = len(predicted_gdf)
print(f"Removed {points_before - points_after} extreme outlier points.")
print(f"Final valid points for statistical analysis: {points_after}")

if points_after == 0:
    print("No points passed the threshold filter. Exiting.")
    exit()


csb_array = predicted_gdf[depth_col].values
gebco_array = predicted_gdf['gebco_depth'].values
diff_array = predicted_gdf['difference'].values

mean_error = np.mean(diff_array)
mae = mean_absolute_error(gebco_array, csb_array)
rmse = np.sqrt(mean_squared_error(gebco_array, csb_array))
std_dev = np.std(diff_array)


if len(diff_array) > 1:
    r, p_value = stats.pearsonr(csb_array, gebco_array)
    r_str = f"{r:.3f} (p-value: {p_value:.3e})"
else:
    r_str = "N/A (Not enough points)"

ci_95 = stats.norm.interval(0.95, loc=mean_error, scale=std_dev/np.sqrt(len(diff_array)))

print("\n--- Statistical Analysis Results (Predicted Grid Only) ---")
print(f"Mean Difference (Bias): {mean_error:.3f} meters")
print(f"Mean Absolute Error (MAE): {mae:.3f} meters")
print(f"Root Mean Square Error (RMSE): {rmse:.3f} meters")
print(f"Standard Deviation of Diff: {std_dev:.3f} meters")
print(f"Pearson Correlation (r): {r:.3f} (p-value: {p_value:.3e})")
print(f"95% Confidence Interval of Mean Diff: {ci_95[0]:.3f} to {ci_95[1]:.3f} meters")

output_gpkg = 'csb_vs_gebco_predicted.gpkg'
predicted_gdf.to_file(output_gpkg, driver="GPKG")
print(f"\nSaved predicted-only points and stats to '{output_gpkg}'")