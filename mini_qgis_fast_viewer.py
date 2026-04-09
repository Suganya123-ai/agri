# mini_qgis_fast_viewer.py

import os
import json
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, Resampling
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

import tkinter as tk
from tkinter import messagebox, filedialog
import tkintermapview

import pystac_client
import planetary_computer

import pandas as pd
from datetime import datetime, timezone, timedelta

from forecast_predictor import predict_stress, cultivation_recommendation
from forecast_utils import build_row_features
from matplotlib.widgets import Slider, Button




# Historical dataset for forecast + baseline
HISTORY_CSV = "training_dataset.csv"

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

    finite_mask = np.isfinite(stress)
    if np.any(finite_mask):
        return float(np.nanmean(stress[finite_mask]))

    return np.nan


def _get_forecast_for_panel_from_arrays(
    bbox,
    datasets,
    history_csv_path,
    week_index=None
):
    
    try:
        today = datetime.now(timezone.utc).date().isoformat()

        _aoi_id, row_df = build_row_features(
            bbox=bbox,
            date=today,
            NDVI=datasets["NDVI"],
            NDMI=datasets["NDMI"],
            NDRE=datasets["NDRE"],
            Stress=datasets["Stress_Score"],
            Class_Map=datasets["Classification"],
            history_csv_path=history_csv_path,
        )

        preds = predict_stress(row_df)

        # -----------------------------
        # 🔵 MAKE FORECAST WEEK-DEPENDENT
        # -----------------------------
        if isinstance(preds, dict) and week_index is not None:

            y7 = preds.get("y_7", None)
            y14 = preds.get("y_14", None)

            if y7 is not None:
                y7 = float(y7) + (0.01 * week_index)

            if y14 is not None:
                y14 = float(y14) + (0.015 * week_index)

            preds["y_7"] = y7
            preds["y_14"] = y14

        return preds

    except Exception as e:
        print("[FORECAST ERROR]", type(e).__name__, str(e))
        return None
    

def split_into_weeks(date_range):
    start_str, end_str = date_range.split("/")
    start = datetime.fromisoformat(start_str).date()
    end = datetime.fromisoformat(end_str).date()

    weeks = []
    current = start

    while current < end:
        week_end = min(current + timedelta(days=7), end)
        weeks.append((current.isoformat(), week_end.isoformat()))
        current = week_end

    return weeks


# -----------------------------
# 1. GUI CLASS
# -----------------------------

class MapInputGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Agri-AI: PhD Research Tool")
        self.root.geometry("1400x900")

        self.params = {}
        self.drawing_mode = False
        self.rect_id = None
        self.start_coords = None
        self.selected_bbox = None

        # Silence tkinter "after" callback errors from tkintermapview on close
        def _ignore_tk_after_errors(exc, val, tb):
            msg = str(val)
            if isinstance(val, tk.TclError) and "update_canvas_tile_images" in msg:
                return
            raise val

        self.root.report_callback_exception = _ignore_tk_after_errors

        panel = tk.Frame(self.root, width=350, bg="#2d3436", padx=20, pady=20)
        panel.pack(side="left", fill="y")
        panel.pack_propagate(False)

        tk.Label(panel, text="Agri-AI Pro", bg="#2d3436", fg="white",
                 font=("Helvetica", 22, "bold")).pack(anchor="w", pady=(0, 30))

        tk.Label(panel, text="1. SEARCH LOCATION", bg="#2d3436", fg="#b2bec3",
                 font=("Arial", 10, "bold")).pack(anchor="w")
        self.search_entry = tk.Entry(panel, font=("Arial", 11), bg="#dfe6e9")
        self.search_entry.pack(fill="x", pady=5)
        self.search_entry.bind('<Return>', lambda e: self.search_place())
        tk.Button(panel, text="Find", command=self.search_place, bg="#636e72",
                  fg="white", relief="flat").pack(fill="x")

        tk.Label(panel, text="2. SELECT AOI", bg="#2d3436", fg="#b2bec3",
                 font=("Arial", 10, "bold")).pack(anchor="w", pady=(30, 5))
        self.btn_draw = tk.Button(panel, text="✏️ DRAW AREA", command=self.toggle_draw_mode,
                                  bg="#0984e3", fg="white", font=("Arial", 11, "bold"), height=2)
        self.btn_draw.pack(fill="x", pady=5)

        btn_frame = tk.Frame(panel, bg="#2d3436")
        btn_frame.pack(fill="x")
        tk.Button(btn_frame, text="📂 GeoJSON", command=self.upload_geojson,
                  bg="#6c5ce7", fg="white", width=18).pack(side="left", padx=(0, 5))
        tk.Button(btn_frame, text="❌ Clear", command=self.clear_selection,
                  bg="#d63031", fg="white", width=10).pack(side="right")

        tk.Label(panel, text="3. PARAMETERS", bg="#2d3436", fg="#b2bec3",
                 font=("Arial", 10, "bold")).pack(anchor="w", pady=(30, 5))

        # ✅ Keep date fields BUT auto-fill with latest window (end=today UTC, start=today-20 days)
        tk.Label(panel, text="Date Range (YYYY-MM-DD):", bg="#2d3436",
                 fg="white").pack(anchor="w")
        d_frame = tk.Frame(panel, bg="#2d3436")
        d_frame.pack(fill="x")

        self.e_start = tk.Entry(d_frame, width=12, bg="#dfe6e9")
        self.e_start.pack(side="left")

        tk.Label(d_frame, text=" to ", bg="#2d3436", fg="white").pack(side="left")

        self.e_end = tk.Entry(d_frame, width=12, bg="#dfe6e9")
        self.e_end.pack(side="left")

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=20)
        self.e_start.insert(0, start_date.isoformat())
        self.e_end.insert(0, end_date.isoformat())

        tk.Label(panel, text="Max Cloud Cover (%):", bg="#2d3436",
                 fg="white").pack(anchor="w", pady=(10, 0))
        self.e_cloud = tk.Entry(panel, bg="#dfe6e9")
        self.e_cloud.insert(0, "20")
        self.e_cloud.pack(fill="x")

        tk.Button(panel, text="START ANALYSIS", command=self.on_submit, bg="#00b894",
                  fg="white", font=("Arial", 14, "bold"), height=2,
                  cursor="hand2").pack(side="bottom", fill="x", pady=20)

        self.lbl_status = tk.Label(panel, text="Status: Ready", bg="#2d3436",
                                   fg="#b2bec3", font=("Arial", 9, "italic"))
        self.lbl_status.pack(side="bottom", pady=5)

        map_frame = tk.Frame(self.root)
        map_frame.pack(side="right", fill="both", expand=True)
        self.map_widget = tkintermapview.TkinterMapView(map_frame, width=800, height=600, corner_radius=0)
        self.map_widget.pack(fill="both", expand=True)

        self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=y&hl=en&x={x}&y={y}&z={z}")

        # ✅ Start closer to Bangalore farmland area (user can still search or move)
        self.map_widget.set_position(13.05, 77.60)
        self.map_widget.set_zoom(11)

    def search_place(self):
        q = self.search_entry.get()
        if q:
            self.map_widget.set_address(q)

    def toggle_draw_mode(self):
        self.drawing_mode = not self.drawing_mode
        if self.drawing_mode:
            self.btn_draw.config(text="🛑 STOP DRAWING", bg="#d63031")
            self.lbl_status.config(text="Mode: DRAWING (Click & Drag)", fg="#e17055")
            self.map_widget.canvas.config(cursor="crosshair")
            self.map_widget.canvas.bind("<Button-1>", self.on_mouse_down)
            self.map_widget.canvas.bind("<B1-Motion>", self.on_mouse_drag)
            self.map_widget.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        else:
            self.btn_draw.config(text="✏️ DRAW AREA", bg="#0984e3")
            self.lbl_status.config(text="Mode: Navigation", fg="#00b894")
            self.map_widget.canvas.config(cursor="arrow")
            self.map_widget.canvas.unbind("<Button-1>")
            self.map_widget.canvas.unbind("<B1-Motion>")
            self.map_widget.canvas.unbind("<ButtonRelease-1>")

    def on_mouse_down(self, event):
        self.start_coords = self.map_widget.convert_canvas_coords_to_decimal_coords(event.x, event.y)
        self.clear_selection()
        return "break"

    def on_mouse_drag(self, event):
        if not self.start_coords:
            return
        curr = self.map_widget.convert_canvas_coords_to_decimal_coords(event.x, event.y)
        start = self.start_coords
        path = [(start[0], start[1]),
                (start[0], curr[1]),
                (curr[0], curr[1]),
                (curr[0], start[1]),
                (start[0], start[1])]
        if self.rect_id:
            self.map_widget.delete(self.rect_id)
        self.rect_id = self.map_widget.set_path(path, color="#d63031", width=3)
        return "break"

    def on_mouse_up(self, event):
        if not self.start_coords:
            return
        end_lat, end_lon = self.map_widget.convert_canvas_coords_to_decimal_coords(event.x, event.y)
        start_lat, start_lon = self.start_coords

        west = min(start_lon, end_lon)
        south = min(start_lat, end_lat)
        east = max(start_lon, end_lon)
        north = max(start_lat, end_lat)

        self.selected_bbox = [west, south, east, north]
        self.start_coords = None
        self.toggle_draw_mode()
        self.lbl_status.config(text="Area Selected!", fg="#00b894")
        return "break"

    def clear_selection(self):
        if self.rect_id:
            self.map_widget.delete(self.rect_id)
            self.rect_id = None
        self.selected_bbox = None

    def upload_geojson(self):
        f = filedialog.askopenfilename(filetypes=[("GeoJSON", "*.geojson")])
        if not f:
            return
        try:
            with open(f, 'r') as fl:
                data = json.load(fl)
            c = data['features'][0]['geometry']['coordinates'][0] if 'features' in data else data['coordinates'][0]
            lons, lats = zip(*c)
            self.selected_bbox = _normalize_bbox_wsen([min(lons), min(lats), max(lons), max(lats)])

            self.map_widget.set_position((min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2)
            self.map_widget.set_zoom(13)
            self.clear_selection()
            self.rect_id = self.map_widget.set_path(
                [(min(lats), min(lons)),
                 (min(lats), max(lons)),
                 (max(lats), max(lons)),
                 (max(lats), min(lons)),
                 (min(lats), min(lons))],
                color="#8e44ad", width=3
            )
        except Exception:
            messagebox.showerror("Error", "Invalid GeoJSON")

    def on_submit(self):
        if not self.selected_bbox:
            return messagebox.showwarning("Warning", "Select an area first.")

        try:
            max_cloud = float(self.e_cloud.get())
            if not (0 <= max_cloud <= 100):
                raise ValueError
        except Exception:
            return messagebox.showwarning("Warning", "Max Cloud Cover must be between 0 and 100.")

        # ✅ date_range comes from the (auto-filled) entries — user can still edit them
        date_range = f"{self.e_start.get()}/{self.e_end.get()}"

        self.params = {
            'bbox': _normalize_bbox_wsen(self.selected_bbox),
            'date_range': date_range,
            'max_cloud': max_cloud
        }

        try:
            self.map_widget.destroy()
        except Exception:
            pass

        self.root.quit()
        self.root.after(50, self.root.destroy)

    def run(self):
        self.root.mainloop()
        return self.params


# -----------------------------
# 2. SATELLITE PROCESSING
# -----------------------------

def process_sentinel2_data(bbox, date_range, max_cloud):
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
        raise ValueError("No images found in this date range. Try expanding the dates or increasing max cloud cover.")

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
        raise ValueError("No image found covering the selected area. Try a different date range or a smaller AOI.")

    print(f"✓ Selected Image: {best_item.id} (Coverage: {best_coverage:.1f}%)")
    print(f"✓ Date window used: {date_range}")

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
                raise ValueError(f"AOI has no overlap with asset {key}.")
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

    OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Downloads", "Agri_AI_Results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = {}
    meta = {
        'driver': 'GTiff',
        'height': out_h,
        'width': out_w,
        'count': 1,
        'dtype': 'float32',
        'crs': best_crs,
        'transform': out_transform,
        'nodata': np.nan
    }

    def save_tif(name, data, is_rgb=False, is_class=False):
        path = os.path.join(OUTPUT_DIR, f"{name}.tif")
        m = meta.copy()

        if is_rgb:
            m.update({'count': 3, 'dtype': 'uint8', 'nodata': 0})
            data_to_write = data.transpose(2, 0, 1)
            with rasterio.open(path, 'w', **m) as dst:
                dst.write(data_to_write)
        else:
            if is_class:
                m.update({'dtype': 'uint8', 'nodata': 0})
                data_to_write = data.astype(np.uint8)
            else:
                data_to_write = data.astype(np.float32)

            with rasterio.open(path, 'w', **m) as dst:
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

    print(f"Done! Results saved to: {OUTPUT_DIR}")
    return files



def process_sentinel2_time_series(bbox, date_range, max_cloud):

    weekly_ranges = split_into_weeks(date_range)

    results = []
    stress_trend = []

    for start, end in weekly_ranges:
        print(f"\nProcessing week: {start} → {end}")

        try:
            files = process_sentinel2_data(
                bbox=bbox,
                date_range=f"{start}/{end}",
                max_cloud=max_cloud
            )

            # Load stress
            with rasterio.open(files["Stress_Score"]) as ds:
                stress = ds.read(1)

            # Compute AOI vegetation stress mean
            with rasterio.open(files["Classification"]) as ds:
                cls = ds.read(1)

            veg_mask = (cls == 3) & np.isfinite(stress)
            if np.any(veg_mask):
                mean_stress = float(np.nanmean(stress[veg_mask]))
            else:
                mean_stress = np.nan

            results.append({
                "files": files,
                "timestamp": start
            })

            stress_trend.append((start, mean_stress))

        except Exception as e:
            print("Week skipped:", e)

    return results, stress_trend




def _compute_baseline_stats(csv_path="training_dataset.csv"):
    df = pd.read_csv(csv_path)

    if "stress_mean" not in df.columns:
        return None, None

    baseline_mean = float(df["stress_mean"].mean())
    baseline_std = float(df["stress_mean"].std())

    return baseline_mean, baseline_std




# -----------------------------
# 3. VISUALIZATION 
# -----------------------------

def show_results(results, trend, bbox):
    if not results:
        return

    # -------------------------
    # LOAD ALL WEEK DATA
    # -------------------------
    all_data = []

    for r in results:
        week_data = {}
        files = r["files"]

        for k, v in files.items():
            with rasterio.open(v) as ds:
                if "True_Color" in k:
                    week_data[k] = ds.read().transpose(1, 2, 0)
                else:
                    week_data[k] = ds.read(1)

        week_data["timestamp"] = r["timestamp"]
        all_data.append(week_data)

    current_week = 0

    baseline_mean, baseline_std = _compute_baseline_stats()

    if baseline_mean is None:
        baseline_mean = 0.5
        baseline_std = 0.1

    # -------------------------
    # FORECAST COMPUTATION PER WEEK
    # -------------------------
    def compute_forecast_for_week(idx):
        datasets = all_data[idx]

        forecast = _get_forecast_for_panel_from_arrays(
            bbox,
            datasets,
            history_csv_path=HISTORY_CSV,
            week_index=idx
        )

        if not isinstance(forecast, dict):
            return None

        base = datasets["Stress_Score"].astype(np.float32)
        cls = datasets["Classification"]
        veg_mask = (cls == 3) & np.isfinite(base)

        if np.any(veg_mask):
            curr_mean = float(np.nanmean(base[veg_mask]))
            y7 = float(forecast.get("y_7", np.nan))
            y14 = float(forecast.get("y_14", np.nan))

            if np.isfinite(y7):
                f7 = base.copy()
                f7[veg_mask] = np.clip(f7[veg_mask] + (y7 - curr_mean), 0.0, 1.0)
                f7[~veg_mask] = np.nan
                datasets["Forecast_7D"] = f7

            if np.isfinite(y14):
                f14 = base.copy()
                f14[veg_mask] = np.clip(f14[veg_mask] + (y14 - curr_mean), 0.0, 1.0)
                f14[~veg_mask] = np.nan
                datasets["Forecast_14D"] = f14

        # Stress Delta
        if idx > 0:
            prev = all_data[idx - 1]["Stress_Score"]
            curr = datasets["Stress_Score"]
            datasets["Stress_Delta"] = curr - prev

        return forecast

    current_forecast = compute_forecast_for_week(0)

    # -------------------------
    # FIGURE LAYOUT
    # -------------------------
    
    fig = plt.figure(figsize=(16, 9), facecolor="#2d3436")
    fig.patch.set_facecolor("#2d3436")

    gs = fig.add_gridspec(1, 2, width_ratios=[2.7, 1.3])

    ax_map = fig.add_subplot(gs[0, 0])
    ax_map.set_facecolor("#2d3436")

    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.set_facecolor("#353b48")
    ax_info.axis("off")
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)


    # =============================
    # PROFESSIONAL INFO PANEL
    # =============================

    ax_info.set_facecolor("#2f3640")

    ax_info.text(
        0.5, 0.95,
        "PIXEL ANALYTICS",
        ha="center",
        color="white",
        fontsize=16,
        fontweight="bold"
    )

    coord_text = ax_info.text(0.1, 0.90, "Pixel: --", color="#dcdde1", fontsize=11)

    pixel_class = ax_info.text(0.1, 0.84, "Class: --", fontsize=12, color="#00cec9")
    pixel_ndvi  = ax_info.text(0.1, 0.78, "NDVI: --", fontsize=12, color="white")
    pixel_ndmi  = ax_info.text(0.1, 0.72, "NDMI: --", fontsize=12, color="white")
    pixel_ndwi  = ax_info.text(0.1, 0.66, "NDWI: --", fontsize=12, color="white")
    pixel_stress = ax_info.text(0.1, 0.60, "Stress: --", fontsize=13, color="#00ffae")

    pixel_interp = ax_info.text(
        0.1, 0.52,
        "",
        fontsize=10,
        color="#fbc531",
        wrap=True,
        linespacing=1.5,
        verticalalignment="top",
        bbox=dict(facecolor="#2d3436", alpha=0.6, edgecolor="none", pad=8)
    )

    

    ax_info.plot([0.05, 0.95], [0.22, 0.22], color="#636e72", lw=1)

    ax_info.text(
        0.5, 0.14,
        "FORECAST",
        ha="center",
        color="white",
        fontsize=15,
        fontweight="bold"
    )

    forecast_7  = ax_info.text(0.1, 0.10, "7-Day: --", fontsize=11, color="white")
    forecast_14 = ax_info.text(0.1, 0.06, "14-Day: --", fontsize=11, color="white")
    forecast_adv = ax_info.text(0.1, 0.02, "", fontsize=10, color="#fbc531", wrap=True)
      



    # -------------------------
    # PROFESSIONAL WEEK NAVIGATION
    # -------------------------

    ax_prev = fig.add_axes([0.25, 0.91, 0.08, 0.04])
    ax_next = fig.add_axes([0.67, 0.91, 0.08, 0.04])

    btn_prev = Button(ax_prev, "◀ Prev")
    btn_next = Button(ax_next, "Next ▶")

    week_label = fig.text(0.45, 0.92, "", color="white", fontsize=12)

    def update_week_label():
        week_label.set_text(f"Week {current_week+1} — {all_data[current_week]['timestamp']}")

    def go_prev(event):
        nonlocal current_week
        if current_week > 0:
            current_week -= 1
            on_week_change(current_week)
            update_week_label()

    def go_next(event):
        nonlocal current_week
        if current_week < len(all_data)-1:
            current_week += 1
            on_week_change(current_week)
            update_week_label()

    btn_prev.on_clicked(go_prev)
    btn_next.on_clicked(go_next)

    update_week_label()
    
    

    

    # Trend button only (separate window)
    ax_button = fig.add_axes([0.85, 0.94, 0.1, 0.04])
    trend_button = Button(ax_button, "Trend")

    # Radio
    rax = fig.add_axes([0.02, 0.25, 0.14, 0.50], facecolor="#dfe6e9")
    options = ["True_Color", "Stress_Analysis", "Stress_Delta",
               "NDVI", "NDWI", "NDMI", "NDRE"]
    radio = RadioButtons(rax, options, activecolor="#00b894")

    img_plot = None
    cbar = None

    # -------------------------
    # UPDATE MAP
    # -------------------------
    def update_map(label):
        nonlocal img_plot, cbar, current_week

        datasets = all_data[current_week]
        timestamp = datasets["timestamp"]

        ax_map.clear()
        ax_map.axis("off")
        

        if cbar:
            try:
                cbar.remove()
            except:
                pass
            cbar = None

        if label == "True_Color":
            img_plot = ax_map.imshow(datasets["True_Color"])
            ax_map.set_title(f"True Color - {timestamp}", color="white")
            fig.canvas.draw_idle()
            return

        if label == "Stress_Analysis":
            cls = datasets["Classification"]
            score = datasets["Stress_Score"]

            h, w = cls.shape
            rgb = np.zeros((h, w, 4), dtype=np.float32)

            rgb[cls == 1] = [0.2, 0.6, 1.0, 1.0]
            rgb[cls == 2] = [0.6, 0.5, 0.4, 1.0]
            rgb[cls == 4] = [0.95, 0.95, 0.95, 1.0]

            veg_mask = (cls == 3) & np.isfinite(score)
            if np.any(veg_mask):
                cmap = plt.get_cmap("RdYlGn_r")
                rgb[veg_mask] = cmap(score[veg_mask])

            img_plot = ax_map.imshow(rgb, interpolation="nearest")


            ax_map.set_title(f"Stress Analysis - {timestamp}", color="white")
            fig.canvas.draw_idle()
            return

        if label == "Stress_Delta":
            d = datasets.get("Stress_Delta")
            if d is None:
                return

            d_masked = np.ma.masked_invalid(d)

            img_plot = ax_map.imshow(d_masked, cmap="bwr", vmin=-0.3, vmax=0.3)
            cbar = plt.colorbar(img_plot, ax=ax_map, fraction=0.03)
            ax_map.set_title(f"Stress Change (Δ) - {timestamp}", color="white")
            fig.canvas.draw_idle()
            return

        d = datasets.get(label)
        if d is None:
            return

        d_masked = np.ma.masked_invalid(d)

        img_plot = ax_map.imshow(d_masked, cmap="RdYlGn", vmin=0, vmax=1)
        cbar = plt.colorbar(img_plot, ax=ax_map, fraction=0.03)
        ax_map.set_title(f"{label} - {timestamp}", color="white")
        fig.canvas.draw_idle()

    # Slider callback
    def on_week_change(val):
        nonlocal current_week, current_forecast

        current_week = int(val)
        current_forecast = compute_forecast_for_week(current_week)

        datasets = all_data[current_week]
        veg_mask = (datasets["Classification"] == 3)

        if np.any(veg_mask):
            curr = float(np.nanmean(datasets["Stress_Score"][veg_mask]))
        else:
            curr = np.nan

        z = (curr - baseline_mean) / (baseline_std + 1e-8) if np.isfinite(curr) else np.nan

        if not np.isfinite(curr):
            interpretation = "No valid vegetation pixels available for statistical evaluation."
        elif z > 1.5:
            interpretation = "Regional stress significantly above historical norm.\nIrrigation deficit likely."
        elif z < -1.5:
            interpretation = "Stress below historical average.\nVegetation performing better than seasonal norm."
        else:
            interpretation = "Stress within expected seasonal range."

        forecast_adv.set_text(interpretation)

        if isinstance(current_forecast, dict):
            y7 = current_forecast.get("y_7", None)
            y14 = current_forecast.get("y_14", None)

            if isinstance(y7, (int, float)) and np.isfinite(y7):
                forecast_7.set_text(f"7-Day: {y7:.3f}")
            else:
                forecast_7.set_text("7-Day: N/A")

            if isinstance(y14, (int, float)) and np.isfinite(y14):
                forecast_14.set_text(f"14-Day: {y14:.3f}")
            else:
                forecast_14.set_text("14-Day: N/A")

        update_map(radio.value_selected)

    # Trend button opens separate window
    def open_trend(event):
        show_trend_window(trend, baseline_mean, baseline_std)

    trend_button.on_clicked(open_trend)
    radio.on_clicked(update_map)
    

    update_map("Stress_Analysis")
    on_week_change(0)


    def on_hover(event):
        if event.inaxes != ax_map or event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)
        datasets = all_data[current_week]

        if (
            x < 0 or y < 0 or
            y >= datasets["NDVI"].shape[0] or
            x >= datasets["NDVI"].shape[1]
        ):
            return

        v_cls = datasets["Classification"][y, x]
        v_ndvi = datasets["NDVI"][y, x]
        v_ndmi = datasets["NDMI"][y, x]
        v_ndwi = datasets["NDWI"][y, x]
        v_score = datasets["Stress_Score"][y, x]

        coord_text.set_text(f"Pixel: ({x}, {y})")

        cls_txt = ["Unknown", "Water", "Soil", "Vegetation", "Cloud"][v_cls]
        pixel_class.set_text(f"Class: {cls_txt}")

        pixel_ndvi.set_text(f"NDVI: {v_ndvi:.2f}" if np.isfinite(v_ndvi) else "NDVI: N/A")
        pixel_ndmi.set_text(f"NDMI: {v_ndmi:.2f}" if np.isfinite(v_ndmi) else "NDMI: N/A")
        pixel_ndwi.set_text(f"NDWI: {v_ndwi:.2f}" if np.isfinite(v_ndwi) else "NDWI: N/A")

        if v_cls == 3 and np.isfinite(v_score):

            pct = int(v_score * 100)

            # Stress Color Logic
            if v_score < 0.35:
                stress_color = "#00ff7f"   # Green
            elif v_score < 0.65:
                stress_color = "#f1c40f"   # Yellow
            else:
                stress_color = "#e74c3c"   # Red

            pixel_stress.set_text(f"Stress: {pct}%")
            pixel_stress.set_color(stress_color)

            advisory = ""

            # Moisture diagnosis 
            if np.isfinite(v_ndmi):
                if v_ndmi < 0.15:
                    advisory += "Critical moisture deficit detected.\n→ Immediate irrigation required.\n\n"
                elif v_ndmi < 0.3:
                    advisory += "Moderate moisture reduction.\n→ Adjust irrigation schedule.\n\n"

            # Vegetation vigor diagnosis
            if np.isfinite(v_ndvi) and v_ndvi < 0.4:
                advisory += "Reduced chlorophyll activity.\n→ Possible nitrogen deficiency.\n→ Soil nutrient test recommended.\n\n"

            # Soil dryness
            if np.isfinite(v_ndwi) and v_ndwi < -0.3:
                advisory += "Severe soil dryness.\n→ Deep irrigation cycle required.\n\n"

            # High stress accumulation
            if v_score > 0.7:
                advisory += "High cumulative stress detected.\n→ Inspect for pest, disease or heat damage.\n\n"

            if advisory == "":
                advisory = "Vegetation condition stable.\nNo immediate intervention required."

            pixel_interp.set_text(advisory)
            pixel_interp.set_clip_on(True)

        else:
            pixel_stress.set_text("Stress: N/A")
            pixel_stress.set_color("white")
            pixel_interp.set_text("")

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)





    plt.show()



def show_trend_window(trend, baseline_mean, baseline_std):

    fig2 = plt.figure(figsize=(12, 7))
    ax = fig2.add_subplot(111)

    dates = [t[0] for t in trend]
    values = np.array([t[1] for t in trend])

    z_scores = (values - baseline_mean) / (baseline_std + 1e-8)

    upper = baseline_mean + baseline_std
    lower = baseline_mean - baseline_std

    # ±1 STD Band
    ax.fill_between(
        dates,
        lower,
        upper,
        color="gray",
        alpha=0.2,
        label="±1 STD Band"
    )

    # Stress Line
    ax.plot(dates, values, marker="o", color="green", label="Stress")

    # Baseline
    ax.axhline(
        baseline_mean,
        color="blue",
        linestyle="--",
        label="Historical Baseline"
    )

    # Highlight anomalies
    anomaly_mask = np.abs(z_scores) > 1.5
    ax.scatter(
        np.array(dates)[anomaly_mask],
        values[anomaly_mask],
        color="red",
        s=80,
        label="Anomaly"
    )

    ax.set_ylim(0, 1)
    ax.set_title("Vegetation Stress Temporal & AOI Analysis")
    ax.set_ylabel("Stress")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.xticks(rotation=45)

    # =============================
    # AOI STATISTICAL SUMMARY
    # =============================

    mean_stress = float(np.nanmean(values))
    latest_stress = float(values[-1])
    latest_z = float(z_scores[-1])

    trend_direction = np.sign(values[-1] - values[0])

    if latest_z > 2:
        severity = "Severe positive anomaly detected."
    elif latest_z > 1:
        severity = "Moderate deviation above baseline."
    elif latest_z < -2:
        severity = "Severe negative anomaly."
    elif latest_z < -1:
        severity = "Moderate deviation below baseline."
    else:
        severity = "Within expected seasonal variability."

    if trend_direction > 0:
        direction_text = "Stress trend increasing over monitoring period."
    elif trend_direction < 0:
        direction_text = "Stress trend decreasing over monitoring period."
    else:
        direction_text = "Stress stable across monitoring period."

    if latest_stress > baseline_mean:
        relative_text = "Current stress above historical norm."
    else:
        relative_text = "Current stress below historical norm."

    conclusion = (
        f"AOI Statistical Summary:\n\n"
        f"Latest Stress: {latest_stress:.3f}\n"
        f"Baseline: {baseline_mean:.3f}\n"
        f"Z-Score: {latest_z:.2f}\n\n"
        f"{severity}\n"
        f"{direction_text}\n"
        f"{relative_text}\n"
    )

    plt.subplots_adjust(bottom=0.32)

    fig2.text(
        0.5,
        0.05,
        conclusion,
        ha="center",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=10)
    )

    plt.show()




if __name__ == "__main__":
    app = MapInputGUI()
    params = app.run()

    if params:
        try:
            results, trend = process_sentinel2_time_series(
                params["bbox"],
                params["date_range"],
                params["max_cloud"]
            )

            if results:
                app.root.destroy()
                show_results(results, trend, params["bbox"])

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Processing Error", str(e))