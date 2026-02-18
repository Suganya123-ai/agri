import json
import numpy as np
import pandas as pd
from datetime import timedelta

from forecast_utils import (
    normalize_bbox_wsen,
    bbox_centroid_latlon,
    fetch_power_daily,
    iter_dates,
    build_row_features,
    add_targets_nearest,
)

from mini_qgis_fast_viewer import load_arrays_from_sentinel2


def build_training_dataset(
    aois,
    start_date,
    end_date,
    max_cloud=80,
    step_days=10,
    window_days=5,
    min_valid_veg_pixels=200,
    min_cloud_free_ratio=0.20,
    target_tolerance_days=3,
):
    """
    Returns a pandas DataFrame with:
      - AOI/date features from Sentinel-2 maps
      - Weather features from NASA POWER
      - Targets: y_7, y_14
    """
    rows = []

    # Debug counters
    total_queries = 0
    found_arrays = 0
    skipped_low_valid = 0
    skipped_low_cloudfree = 0

    # Prefetch weather for each AOI
    weather_cache = {}
    for aoi in aois:
        aoi_id = aoi["aoi_id"]
        bbox = normalize_bbox_wsen(aoi["bbox"])
        lat, lon = bbox_centroid_latlon(bbox)
        print(f"[WEATHER] Fetching NASA POWER for {aoi_id} at (lat={lat:.5f}, lon={lon:.5f}) ...")
        weather_cache[aoi_id] = fetch_power_daily(lat, lon, start_date, end_date)

    # Loop AOIs and dates
    for aoi in aois:
        aoi_id = aoi["aoi_id"]
        bbox = normalize_bbox_wsen(aoi["bbox"])

        print(f"\n[AOI] {aoi_id} bbox={bbox}")

        for d in iter_dates(start_date, end_date, step_days=step_days):
            total_queries += 1

            # Use a multi-day window to increase chance of finding a usable acquisition
            date_range = f"{d.isoformat()}/{(d + timedelta(days=window_days)).isoformat()}"

            arrays = load_arrays_from_sentinel2(bbox, date_range, max_cloud)
            if arrays is None:
                continue
            found_arrays += 1

            NDVI, NDMI, NDRE, Stress, Class_Map = arrays

            # Basic quality checks
            cloud = (Class_Map == 4)
            cloud_free_ratio = float(np.sum(~cloud) / max(1, Class_Map.size))
            if cloud_free_ratio < min_cloud_free_ratio:
                skipped_low_cloudfree += 1
                continue

            valid_veg = (Class_Map == 3) & np.isfinite(Stress)
            if int(valid_veg.sum()) < min_valid_veg_pixels:
                skipped_low_valid += 1
                continue

            row = build_row_features(d, NDVI, NDMI, NDRE, Stress, Class_Map)
            row["aoi_id"] = aoi_id
            rows.append(row)

    if len(rows) == 0:
        print("\n[DEBUG] No rows created.")
        print("[DEBUG] total_queries:", total_queries)
        print("[DEBUG] found_arrays:", found_arrays)
        print("[DEBUG] skipped_low_cloudfree:", skipped_low_cloudfree)
        print("[DEBUG] skipped_low_valid:", skipped_low_valid)
        raise ValueError(
            "No rows created. Try: larger date range, max_cloud=90, window_days=10, step_days=15, "
            "or verify AOI bbox order [west,south,east,north]."
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    # Merge weather by date for each AOI
    merged_all = []
    for aoi_id, g in df.groupby("aoi_id"):
        wdf = weather_cache[aoi_id]
        gg = g.set_index("date").sort_index()
        merged = gg.join(wdf, how="left").reset_index()
        merged_all.append(merged)

    df = pd.concat(merged_all, ignore_index=True)
    df = df.sort_values(["aoi_id", "date"]).reset_index(drop=True)

    # Create targets per AOI series
    out = []
    for aoi_id, g in df.groupby("aoi_id"):
        gg = g.copy()
        gg = add_targets_nearest(
            gg,
            horizon_days=7,
            tolerance_days=target_tolerance_days,
            target_source_col="stress_mean",
        )
        gg = add_targets_nearest(
            gg,
            horizon_days=14,
            tolerance_days=target_tolerance_days,
            target_source_col="stress_mean",
        )
        out.append(gg)

    df = pd.concat(out, ignore_index=True)

    print("\n[DEBUG] Dataset built successfully.")
    print("[DEBUG] total_queries:", total_queries)
    print("[DEBUG] found_arrays:", found_arrays)
    print("[DEBUG] rows_kept:", len(df))
    print("[DEBUG] y_7 non-null:", int(df["y_7"].notna().sum()))
    print("[DEBUG] y_14 non-null:", int(df["y_14"].notna().sum()))

    return df


if __name__ == "__main__":
    # Load AOIs
    with open("aois.json", "r") as f:
        aois = json.load(f)

    if not isinstance(aois, list) or len(aois) == 0:
        raise ValueError("aois.json must contain a non-empty list of AOIs.")

    # You can tune these to your region
    START_DATE = "2024-01-01"
    END_DATE = "2024-12-31"

    df = build_training_dataset(
        aois=aois,
        start_date=START_DATE,
        end_date=END_DATE,
        max_cloud=40,          # be lenient for building dataset
        step_days=10,          # fewer queries, still reasonable
        window_days=5,         # important: widen search window
        min_valid_veg_pixels=200,
        min_cloud_free_ratio=0.20,
        target_tolerance_days=7,
    )

    df.to_csv("training_dataset.csv", index=False)
    print("\nSaved training_dataset.csv")
