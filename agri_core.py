# agri_core.py
# Core (non-GUI) logic for the web API:
# - Sentinel-2 processing (downloads, indices, stress, classification, GeoTIFF outputs)
# - Loading datasets from saved GeoTIFFs
# - Forecast inference (y_7, y_14)
# - Quality metrics + best cultivation week decision

import os
import json
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, Resampling
import pystac_client
import planetary_computer
from datetime import datetime, timezone

from forecast_predictor import predict_stress
from forecast_utils import build_row_features
import numpy as np


# -----------------------------
# Robust helpers
# -----------------------------

def _normalize_bbox_wsen(bbox):
    """
    Ensure bbox is [west, south, east, north] in EPSG:4326.
    Attempts to detect common lat/lon swaps.
    """
    if bbox is None or len(bbox) != 4:
        raise ValueError("bbox must be a list of 4 numbers [w,s,e,n].")

    a, b, c, d = map(float, bbox)
    w, s, e, n = a, b, c, d

    def _valid(w, s, e, n):
        return (-180 <= w <= 180 and -180 <= e <= 180 and
                -90 <= s <= 90 and -90 <= n <= 90 and
                w < e and s < n)

    if _valid(w, s, e, n):
        return [w, s, e, n]

    # Common swap: bbox accidentally [lat_min, lon_min, lat_max, lon_max]
    w2, s2, e2, n2 = b, a, d, c
    if _valid(w2, s2, e2, n2):
        return [w2, s2, e2, n2]

    raise ValueError(f"Invalid bbox coordinates or ordering: {bbox}")


def _intersect_bounds(b1, b2):
    """
    Intersect bounds b1 and b2 where each is (left, bottom, right, top).
    Returns (l,b,r,t) or None if no overlap.
    """
    l = max(b1[0], b2[0])
    b = max(b1[1], b2[1])
    r = min(b1[2], b2[2])
    t = min(b1[3], b2[3])
    if l >= r or b >= t:
        return None
    return (l, b, r, t)


def load_datasets_from_files(files: dict):
    import numpy as np
    import rasterio

    datasets = {}
    for k, v in files.items():
        with rasterio.open(v) as ds:
            if "True_Color" in k:
                datasets[k] = ds.read().transpose(1, 2, 0)  # H,W,3
            else:
                arr = ds.read(1)
                # Convert nodata to NaN for float rasters
                if ds.nodata is not None and np.issubdtype(arr.dtype, np.floating):
                    arr = arr.copy()
                    arr[arr == ds.nodata] = np.nan
                datasets[k] = arr
    return datasets




def _compute_aoi_current_stress(datasets):
    stress = datasets.get("Stress_Score", None)
    cls = datasets.get("Classification", None)
    if stress is None:
        return np.nan

    # Prefer vegetation pixels
    if cls is not None:
        veg_mask = (cls == 3) & np.isfinite(stress)
        if np.any(veg_mask):
            return float(np.nanmean(stress[veg_mask]))

    # Fallback: any finite stress
    finite_mask = np.isfinite(stress)
    if np.any(finite_mask):
        return float(np.nanmean(stress[finite_mask]))

    return np.nan


def _get_forecast_for_panel_from_arrays(bbox, datasets, history_csv="training_dataset.csv"):
    """
    Build row_df from arrays then predict y_7/y_14 using trained models.
    Returns dict: {"y_7": float, "y_14": float} or None
    """
    try:
        today = datetime.now(timezone.utc).date().isoformat()

        _, row_df = build_row_features(
            bbox=bbox,
            date=today,
            NDVI=datasets["NDVI"],
            NDMI=datasets["NDMI"],
            NDRE=datasets["NDRE"],
            Stress=datasets["Stress_Score"],
            Class_Map=datasets["Classification"],
            history_csv_path=history_csv,
        )

        preds = predict_stress(row_df)
        return preds

    except Exception as e:
        print("[FORECAST ERROR]", type(e).__name__, str(e))
        return None


# -----------------------------
# Quality + cultivation decision
# -----------------------------

def _compute_quality_metrics(datasets):
    """
    Simple explainable quality metrics for confidence scoring.
    - veg_ratio: percent of AOI classified as vegetation
    - valid_stress_ratio: percent of vegetation pixels with finite stress value
    - cloud_ratio: percent of AOI classified as cloud/shadow
    """
    cls = datasets.get("Classification", None)
    stress = datasets.get("Stress_Score", None)

    if cls is None or stress is None:
        return {"veg_ratio": 0.0, "valid_stress_ratio": 0.0, "cloud_ratio": 1.0}

    total = cls.size
    veg = (cls == 3)
    cloud = (cls == 4)

    veg_ratio = float(np.sum(veg) / total) if total else 0.0
    cloud_ratio = float(np.sum(cloud) / total) if total else 1.0

    if np.any(veg):
        valid_stress_ratio = float(np.sum(np.isfinite(stress) & veg) / np.sum(veg))
    else:
        valid_stress_ratio = 0.0

    return {
        "veg_ratio": veg_ratio,
        "valid_stress_ratio": valid_stress_ratio,
        "cloud_ratio": cloud_ratio,
    }


def _choose_best_cultivation_week(curr, y7, y14, quality):
    """
    Choose best cultivation week among:
      - This week (current stress)
      - Next week (7D forecast)
      - In two weeks (14D forecast)

    Decision basis:
      1) lower stress => better cultivation conditions (less water/chlorophyll stress)
      2) confidence depends on AOI data quality + how separated the candidates are
    """

    def _finite(x):
        return (x == x) and np.isfinite(x)

    candidates = []
    if _finite(curr):
        candidates.append(("This week", float(curr)))
    if _finite(y7):
        candidates.append(("Next week (7D)", float(y7)))
    if _finite(y14):
        candidates.append(("In two weeks (14D)", float(y14)))

    if not candidates:
        return {
            "best_week": None,
            "confidence": "Low",
            "reason": "No valid current/forecast stress values available."
        }

    best_week, best_val = min(candidates, key=lambda t: t[1])

    q = quality or {}
    veg_ratio = float(q.get("veg_ratio", 0.0))
    valid_ratio = float(q.get("valid_stress_ratio", 0.0))
    cloud_ratio = float(q.get("cloud_ratio", 1.0))

    # quality score: vegetation + valid stress is most important; clouds reduce confidence
    quality_score = 0.5 * veg_ratio + 0.4 * valid_ratio + 0.1 * (1.0 - cloud_ratio)

    vals = [v for _, v in candidates]
    spread = float(max(vals) - min(vals)) if len(vals) > 1 else 0.0

    if quality_score > 0.65 and spread > 0.08:
        conf = "High"
    elif quality_score > 0.45 and spread > 0.04:
        conf = "Medium"
    else:
        conf = "Low"

    reason = (
        f"Selected '{best_week}' because it has the lowest predicted stress ({best_val:.3f}). "
        f"Quality: veg_ratio={veg_ratio:.2f}, valid_stress_ratio={valid_ratio:.2f}, cloud_ratio={cloud_ratio:.2f}. "
        f"Stress spread across candidate weeks={spread:.3f}."
    )

    return {"best_week": best_week, "confidence": conf, "reason": reason}


# -----------------------------
# Sentinel-2 processing (no GUI)
# -----------------------------

def process_sentinel2_data(bbox, date_range, max_cloud, output_dir=None):
    """
    Downloads best Sentinel-2 L2A item covering bbox, computes NDVI/NDWI/NDMI/NDRE,
    builds Stress_Score + Classification, saves GeoTIFFs.

    Returns: dict {layer_name: filepath}
    """
    print("\n--- 1. SEARCHING SATELLITE DATA ---")

    bbox = _normalize_bbox_wsen(bbox)

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        max_items=50
    )
    items = list(search.items())
    if not items:
        raise ValueError("No images found in this date range. Try expanding dates or increasing max cloud cover.")

    print(f"Found {len(items)} candidates. Calculating best coverage...")

    best_item = None
    best_coverage = 0.0

    for item in items:
        try:
            href = planetary_computer.sign(item.assets["SCL"]).href
            with rasterio.open(href) as src:
                user_bounds_crs = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
                inter = _intersect_bounds(
                    user_bounds_crs,
                    (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
                )
                if inter is None:
                    continue

                area_overlap = (inter[2] - inter[0]) * (inter[3] - inter[1])
                area_user = (user_bounds_crs[2] - user_bounds_crs[0]) * (user_bounds_crs[3] - user_bounds_crs[1])
                coverage = (area_overlap / area_user) * 100.0

                if coverage > best_coverage:
                    best_coverage = coverage
                    best_item = item
        except Exception:
            continue

    if not best_item:
        raise ValueError("No image found covering the selected area. Try different dates or a smaller AOI.")

    print(f"✓ Selected Image: {best_item.id} (Coverage: {best_coverage:.1f}%)")

    href_scl = planetary_computer.sign(best_item.assets["SCL"]).href
    with rasterio.open(href_scl) as scl_src:
        best_crs = scl_src.crs
        aoi_bounds_crs = transform_bounds("EPSG:4326", best_crs, *bbox, densify_pts=21)

    href_b4 = planetary_computer.sign(best_item.assets["B04"]).href
    with rasterio.open(href_b4) as src:
        inter_b4 = _intersect_bounds(
            aoi_bounds_crs,
            (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        )
        if inter_b4 is None:
            raise ValueError("Selected image does not overlap AOI for B04 asset (unexpected).")

        window_full = from_bounds(*inter_b4, transform=src.transform)
        width = int(np.ceil(window_full.width))
        height = int(np.ceil(window_full.height))

        MAX_PX = 3000
        scale = 1.0
        if width > MAX_PX or height > MAX_PX:
            scale = MAX_PX / max(width, height)
            print(f"   Optimization: Scaling image by {scale:.2f}x")

        out_w = max(1, int(width * scale))
        out_h = max(1, int(height * scale))
        out_shape = (out_h, out_w)

        win_transform = src.window_transform(window_full)
        out_transform = win_transform * win_transform.scale(width / out_w, height / out_h)

    final_bounds_crs = inter_b4

    def get_band(key, resampling=Resampling.cubic):
        url = planetary_computer.sign(best_item.assets[key]).href
        with rasterio.open(url) as src:
            inter = _intersect_bounds(
                final_bounds_crs,
                (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            )
            if inter is None:
                raise ValueError(f"AOI has no overlap with asset {key}. Try another date or smaller AOI.")

            win = from_bounds(*inter, transform=src.transform)
            data = src.read(
                1,
                window=win,
                out_shape=out_shape,
                fill_value=0,
                resampling=resampling
            )
            return data

    print("Downloading Spectral Bands...")

    B4 = get_band("B04").astype(np.float32) / 10000.0
    B8 = get_band("B08").astype(np.float32) / 10000.0
    B3 = get_band("B03").astype(np.float32) / 10000.0
    B2 = get_band("B02").astype(np.float32) / 10000.0
    B5 = get_band("B05").astype(np.float32) / 10000.0
    B11 = get_band("B11").astype(np.float32) / 10000.0
    B12 = get_band("B12").astype(np.float32) / 10000.0
    SCL = get_band("SCL", resampling=Resampling.nearest).astype(np.uint8)

    print("Calculating Scientific Indices...")
    np.seterr(divide='ignore', invalid='ignore')

    NDVI = (B8 - B4) / (B8 + B4 + 1e-8)
    NDWI = (B3 - B8) / (B3 + B8 + 1e-8)
    NDMI = (B8 - B11) / (B8 + B11 + 1e-8)
    NDRE = (B8 - B5) / (B8 + B5 + 1e-8)

    mask_cloud_shadow = np.isin(SCL, [1, 2, 3, 7, 8, 9, 10, 11])
    for arr in [NDVI, NDMI, NDRE, NDWI]:
        arr[mask_cloud_shadow] = np.nan

    mask_water = (SCL == 6) | (np.nan_to_num(NDWI, nan=-999.0) > 0.15)
    mask_soil = (SCL == 5) | ((NDVI < 0.25) & ~mask_water & ~mask_cloud_shadow)
    mask_veg = (SCL == 4) | ((NDVI >= 0.25) & ~mask_water & ~mask_cloud_shadow)

    Stress_Intensity = np.full_like(NDVI, np.nan, dtype=np.float32)
    s_water = np.clip(1 - ((NDMI - (-0.15)) / (0.3 - (-0.15))), 0, 1)
    s_chl = np.clip(1 - ((NDRE - 0.15) / (0.5 - 0.15)), 0, 1)
    combined_stress = (s_water * 0.6) + (s_chl * 0.4)
    Stress_Intensity[mask_veg] = combined_stress[mask_veg].astype(np.float32)

    rgb_stack = np.dstack((B4, B3, B2))
    rgb_bright = np.clip(rgb_stack * 2.5, 0, 1)
    rgb_uint8 = (rgb_bright * 255).astype(np.uint8)

    if output_dir is None:
        output_dir = os.path.join(os.path.expanduser("~"), "Downloads", "Agri_AI_Results")
    os.makedirs(output_dir, exist_ok=True)

    files = {}
    meta = {
        "driver": "GTiff",
        "height": out_h,
        "width": out_w,
        "count": 1,
        "dtype": "float32",
        "crs": best_crs,
        "transform": out_transform,
        "nodata": np.nan
    }

    def save_tif(name, data, is_rgb=False, is_class=False):
        path = os.path.join(output_dir, f"{name}.tif")
        m = meta.copy()

        if is_rgb:
            m.update({"count": 3, "dtype": "uint8", "nodata": 0})
            data_to_write = data.transpose(2, 0, 1)
            with rasterio.open(path, "w", **m) as dst:
                dst.write(data_to_write)
        else:
            if is_class:
                m.update({"dtype": "uint8", "nodata": 0})
                data_to_write = data.astype(np.uint8)
            else:
                data_to_write = data.astype(np.float32)

            with rasterio.open(path, "w", **m) as dst:
                dst.write(data_to_write, 1)

        files[name] = path

    save_tif("NDVI", NDVI)
    save_tif("NDWI", NDWI)
    save_tif("NDMI", NDMI)
    save_tif("NDRE", NDRE)
    save_tif("Stress_Score", Stress_Intensity)
    save_tif("True_Color", rgb_uint8, is_rgb=True)

    Class_Map = np.zeros_like(NDVI, dtype=np.uint8)
    Class_Map[mask_water] = 1
    Class_Map[mask_soil] = 2
    Class_Map[mask_veg] = 3
    Class_Map[mask_cloud_shadow] = 4
    save_tif("Classification", Class_Map, is_class=True)

    print(f"Done! Results saved to: {output_dir}")
    return files








def compute_pixel_explanation(datasets, x: int, y: int) -> dict:
    cls = int(datasets["Classification"][y, x])
    stress = datasets["Stress_Score"][y, x]
    ndmi = datasets["NDMI"][y, x]
    ndre = datasets["NDRE"][y, x]
    ndvi = datasets["NDVI"][y, x]

    class_name = {1: "Water", 2: "Soil", 3: "Vegetation", 4: "Cloud/Snow"}.get(cls, "Unknown")

    if cls != 3 or not np.isfinite(stress):
        return {
            "class_name": class_name,
            "stress_percent": "N/A",
            "reason": "This point is not vegetation (or is masked by clouds).",
            "solution": "Click on a vegetation area (green field) to get agronomic advice."
        }

    stress_pct = f"{stress*100:.1f}%"

    # Very simple, human-friendly diagnosis
    reasons = []
    actions = []

    if np.isfinite(ndmi) and ndmi < 0.05:
        reasons.append("Low moisture detected")
        actions.append("Increase irrigation or check irrigation system")

    if np.isfinite(ndre) and ndre < 0.20:
        reasons.append("Low chlorophyll / possible nutrient stress")
        actions.append("Check fertilization (especially nitrogen) and soil nutrients")

    if not reasons:
        if stress < 0.35:
            reasons.append("Vegetation looks healthy")
            actions.append("Maintain current irrigation and monitoring")
        elif stress < 0.60:
            reasons.append("Moderate stress detected")
            actions.append("Monitor closely, consider light irrigation if no rain")
        else:
            reasons.append("High stress detected")
            actions.append("Urgent: irrigate and check nutrients + pests")

    return {
        "class_name": class_name,
        "stress_percent": stress_pct,
        "reason": "; ".join(reasons),
        "solution": "; ".join(actions),
    }


def choose_best_cultivation_simple(curr: float, y7: float, y14: float) -> dict:
    # Simple explanation for everyone:
    # “Best week = lower predicted stress”
    if not np.isfinite(y7) and not np.isfinite(y14):
        return {
            "best_week": "Unknown",
            "reason": "Forecast is not available, so we cannot compare weeks.",
            "confidence": "Low"
        }

    if np.isfinite(y7) and not np.isfinite(y14):
        return {
            "best_week": "Next 7 days",
            "reason": "Only the 7-day forecast is available.",
            "confidence": "Medium"
        }

    if np.isfinite(y14) and not np.isfinite(y7):
        return {
            "best_week": "Next 14 days",
            "reason": "Only the 14-day forecast is available.",
            "confidence": "Medium"
        }

    # both exist
    diff = abs(y7 - y14)
    if y7 < y14:
        best = "Next 7 days"
        reason = "Next week has lower predicted stress, so crops should establish more easily."
    elif y14 < y7:
        best = "Next 14 days"
        reason = "Two weeks from now has lower predicted stress, so conditions should be better."
    else:
        best = "Either week"
        reason = "Both weeks have very similar predicted stress."

    if diff >= 0.05:
        conf = "High"
    elif diff >= 0.02:
        conf = "Medium"
    else:
        conf = "Low"

    return {"best_week": best, "reason": reason, "confidence": conf}
