# forecast_utils.py
import hashlib
import numpy as np
import pandas as pd


TARGETS = ["y_7", "y_14"]
LAG_COLS = ["stress_lag_1", "stress_lag_2", "stress_lag_3", "stress_roll_3"]


def bbox_to_aoi_id(bbox):
    """
    Stable AOI id from bbox (rounded -> hash).
    bbox must be [west,south,east,north].
    """
    b = [float(x) for x in bbox]
    b = [round(x, 6) for x in b]
    s = ",".join(map(str, b))
    return "aoi_" + hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def safe_nanmean(a):
    a = np.asarray(a)
    if a.size == 0:
        return np.nan
    return float(np.nanmean(a))


def safe_nanpercentile(a, q):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return float(np.percentile(a, q))


def compute_aoi_summary_from_arrays(NDVI, NDMI, NDRE, Stress, Class_Map):
    """
    Builds the core AOI summary features (similar style to training_dataset.csv).
    We compute means/percentiles on vegetation pixels.
    """
    h, w = Class_Map.shape
    pix_total = int(h * w)

    cloud_mask = (Class_Map == 4)
    water_mask = (Class_Map == 1)
    soil_mask  = (Class_Map == 2)
    veg_mask   = (Class_Map == 3)

    pix_cloud_free = int(np.sum(~cloud_mask))
    pix_veg = int(np.sum(veg_mask))
    pix_valid = pix_veg  # treat vegetation as valid for stress metrics

    pct_cloud_free = pix_cloud_free / pix_total if pix_total else np.nan
    pct_veg = pix_veg / pix_total if pix_total else np.nan
    pct_valid = pix_valid / pix_total if pix_total else np.nan

    # compute stats on veg pixels only
    ndvi_v = NDVI[veg_mask]
    ndmi_v = NDMI[veg_mask]
    ndre_v = NDRE[veg_mask]
    stress_v = Stress[veg_mask]

    # stressed threshold consistent with your earlier mask logic
    stressed_mask = np.isfinite(stress_v) & (stress_v >= 0.3)
    pct_stressed = float(np.sum(stressed_mask) / stress_v.size) if stress_v.size else np.nan

    out = {
        "pix_total": pix_total,
        "pix_cloud_free": pix_cloud_free,
        "pix_veg": pix_veg,
        "pix_valid": pix_valid,
        "pct_cloud_free": pct_cloud_free,
        "pct_veg": pct_veg,
        "pct_valid": pct_valid,
        "pct_stressed": pct_stressed,

        "stress_mean": safe_nanmean(stress_v),
        "stress_p90": safe_nanpercentile(stress_v, 90),

        "ndvi_mean": safe_nanmean(ndvi_v),
        "ndvi_median": safe_nanpercentile(ndvi_v, 50),
        "ndvi_std": float(np.nanstd(ndvi_v)) if np.isfinite(ndvi_v).any() else np.nan,
        "ndvi_p10": safe_nanpercentile(ndvi_v, 10),
        "ndvi_p90": safe_nanpercentile(ndvi_v, 90),

        "ndmi_mean": safe_nanmean(ndmi_v),
        "ndmi_median": safe_nanpercentile(ndmi_v, 50),
        "ndmi_std": float(np.nanstd(ndmi_v)) if np.isfinite(ndmi_v).any() else np.nan,
        "ndmi_p10": safe_nanpercentile(ndmi_v, 10),
        "ndmi_p90": safe_nanpercentile(ndmi_v, 90),

        "ndre_mean": safe_nanmean(ndre_v),
        "ndre_median": safe_nanpercentile(ndre_v, 50),
        "ndre_std": float(np.nanstd(ndre_v)) if np.isfinite(ndre_v).any() else np.nan,
        "ndre_p10": safe_nanpercentile(ndre_v, 10),
        "ndre_p90": safe_nanpercentile(ndre_v, 90),
    }
    return out


def add_lags_from_history(row_dict, history_df):
    """
    history_df must be filtered to this AOI and sorted by date.
    Adds: stress_lag_1/2/3 and stress_roll_3 (using shift(1)).
    """
    if history_df is None or len(history_df) == 0:
        # No history -> fallback: use current stress as lag proxies
        s = row_dict.get("stress_mean", np.nan)
        row_dict["stress_lag_1"] = s
        row_dict["stress_lag_2"] = s
        row_dict["stress_lag_3"] = s
        row_dict["stress_roll_3"] = s
        return row_dict

    h = history_df.sort_values("date").copy()
    # last known stress values
    s_hist = h["stress_mean"].dropna().values
    if s_hist.size == 0:
        s = row_dict.get("stress_mean", np.nan)
        row_dict["stress_lag_1"] = s
        row_dict["stress_lag_2"] = s
        row_dict["stress_lag_3"] = s
        row_dict["stress_roll_3"] = s
        return row_dict

    def get_k_back(k):
        return float(s_hist[-k]) if s_hist.size >= k else float(s_hist[0])

    row_dict["stress_lag_1"] = get_k_back(1)
    row_dict["stress_lag_2"] = get_k_back(2)
    row_dict["stress_lag_3"] = get_k_back(3)

    # roll_3 = mean of last 3 values
    row_dict["stress_roll_3"] = float(np.mean(s_hist[-3:])) if s_hist.size >= 3 else float(np.mean(s_hist))

    return row_dict


def build_row_features(
    bbox,
    date,
    NDVI,
    NDMI,
    NDRE,
    Stress,
    Class_Map,
    history_csv_path="training_dataset.csv",
):
    """
    Returns (aoi_id, row_df) where row_df is a single-row DataFrame
    ready for prediction (contains numeric features + lag features).
    If history exists for this AOI, lags come from history; otherwise fallback.
    """
    aoi_id = bbox_to_aoi_id(bbox)

    base = compute_aoi_summary_from_arrays(NDVI, NDMI, NDRE, Stress, Class_Map)
    base["aoi_id"] = aoi_id
    base["date"] = pd.to_datetime(date)

    # pull history (if file exists)
    hist = None
    try:
        hdf = pd.read_csv(history_csv_path)
        if "date" in hdf.columns:
            hdf["date"] = pd.to_datetime(hdf["date"])
        if "aoi_id" in hdf.columns:
            hist = hdf[hdf["aoi_id"] == aoi_id].copy()
    except Exception:
        hist = None

    base = add_lags_from_history(base, hist)

    # IMPORTANT:
    # Your model was trained with many columns (weather columns may exist).
    # If we don't have them now, leave them missing -> LightGBM can handle NaN.
    row_df = pd.DataFrame([base])

    # Ensure targets NOT present
    for t in TARGETS:
        if t in row_df.columns:
            row_df = row_df.drop(columns=[t])

    return aoi_id, row_df
