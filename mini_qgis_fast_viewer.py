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


def _get_forecast_for_panel_from_arrays(bbox, datasets, history_csv="training_dataset.csv"):
    """
    Build row_df from arrays then predict y_7/y_14 using trained models.
    Returns dict: {"y_7": float, "y_14": float} or None
    """
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
            history_csv_path=history_csv,
        )

        preds = predict_stress(row_df)
        return preds

    except Exception as e:
        print("[FORECAST ERROR]", type(e).__name__, str(e))
        return None


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


# -----------------------------
# 3. VISUALIZATION (with forecast panel + forecast layers)
# -----------------------------

def show_results(files, bbox):
    if not files:
        return

    datasets = {}
    for k, v in files.items():
        if "True_Color" in k:
            with rasterio.open(v) as ds:
                datasets[k] = ds.read().transpose(1, 2, 0)
        else:
            with rasterio.open(v) as ds:
                datasets[k] = ds.read(1)

    forecast = _get_forecast_for_panel_from_arrays(bbox, datasets)
    print("[DEBUG] forecast =", forecast)

    # Build forecast raster layers (Forecast_7D / Forecast_14D)
    try:
        base = datasets["Stress_Score"].astype(np.float32)
        cls = datasets["Classification"]
        veg_mask = (cls == 3) & np.isfinite(base)

        if isinstance(forecast, dict) and np.any(veg_mask):
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

    except Exception as e:
        print("[FORECAST LAYERS ERROR]", type(e).__name__, str(e))

    fig = plt.figure(figsize=(16, 9), facecolor="#2d3436")
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

    ax_map = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.set_facecolor("#353b48")
    ax_info.axis("off")

    rax = plt.axes([0.02, 0.35, 0.14, 0.50], facecolor="#dfe6e9")
    options = ["True_Color", "Stress_Analysis", "NDVI", "NDWI", "NDMI", "NDRE"]
    if "Forecast_7D" in datasets:
        options += ["Forecast_7D", "Forecast_14D"]

    radio = RadioButtons(rax, options, activecolor="#00b894")

    img_plot = None
    cbar = None

    ax_info.text(0.5, 0.95, "PIXEL INSPECTOR", ha="center",
                 color="white", fontsize=18, fontweight="bold")
    coord_text = ax_info.text(0.1, 0.88, "Hover over map...",
                              color="#b2bec3", fontsize=12)

    stat_texts = []
    labels = ["Class", "NDVI", "NDMI (Moist)", "NDWI (Water)", "Stress %"]
    for i, lbl in enumerate(labels):
        t = ax_info.text(0.1, 0.78 - (i * 0.08), f"{lbl}: --",
                         color="white", fontsize=14)
        stat_texts.append(t)

    desc_text = ax_info.text(0.1, 0.35, "", color="#f1c40f",
                             fontsize=11, wrap=True)

    ax_info.text(0.5, 0.22, "LIVE FORECAST (AOI)", ha="center",
                 color="white", fontsize=16, fontweight="bold")

    live_text_current = ax_info.text(0.1, 0.16, "Current Stress: --",
                                     color="white", fontsize=12)
    live_text_y7 = ax_info.text(0.1, 0.12, "7-Day Forecast: --",
                                color="white", fontsize=12)
    live_text_y14 = ax_info.text(0.1, 0.08, "14-Day Forecast: --",
                                 color="white", fontsize=12)
    live_text_adv = ax_info.text(0.1, 0.03, "Advisory: --",
                                 color="#fdcb6e", fontsize=11, wrap=True)

    curr = _compute_aoi_current_stress(datasets)

    if np.isfinite(curr):
        live_text_current.set_text(f"Current Stress: {curr:.4f} ({int(curr*100)}%)")
    else:
        live_text_current.set_text("Current Stress: N/A")

    if isinstance(forecast, dict):
        y7 = float(forecast.get("y_7", np.nan))
        y14 = float(forecast.get("y_14", np.nan))

        live_text_y7.set_text(
            f"7-Day Forecast: {y7:.4f} ({int(y7*100)}%)" if np.isfinite(y7) else "7-Day Forecast: N/A"
        )
        live_text_y14.set_text(
            f"14-Day Forecast: {y14:.4f} ({int(y14*100)}%)" if np.isfinite(y14) else "14-Day Forecast: N/A"
        )

        if np.isfinite(curr) and np.isfinite(y7) and np.isfinite(y14):
            advisory = cultivation_recommendation(curr, y7, y14)
            live_text_adv.set_text(f"Advisory: {advisory}")
        else:
            live_text_adv.set_text("Advisory: N/A (missing values)")
    else:
        live_text_y7.set_text("7-Day Forecast: N/A")
        live_text_y14.set_text("14-Day Forecast: N/A")
        live_text_adv.set_text("Advisory: forecast not available")

    def update_map(label):
        nonlocal img_plot, cbar
        ax_map.clear()
        ax_map.axis("off")

        if cbar:
            try:
                cbar.remove()
            except Exception:
                pass
            cbar = None

        if label == "True_Color":
            img_plot = ax_map.imshow(datasets["True_Color"])
            ax_map.set_title("True Color (RGB)", color="white")
            fig.canvas.draw_idle()
            return

        if label == "Stress_Analysis":
            cls = datasets["Classification"]
            score = datasets["Stress_Score"]
            h, w = cls.shape
            rgb = np.zeros((h, w, 4), dtype=np.float32)

            rgb[cls == 1] = [0.2, 0.6, 1.0, 1.0]        # Water
            rgb[cls == 2] = [0.6, 0.5, 0.4, 1.0]        # Soil
            rgb[cls == 4] = [0.95, 0.95, 0.95, 1.0]     # Cloud

            veg_mask = (cls == 3) & np.isfinite(score)
            if np.any(veg_mask):
                cmap = plt.get_cmap("RdYlGn_r")
                rgb[veg_mask] = cmap(score[veg_mask])

            img_plot = ax_map.imshow(rgb)
            ax_map.set_title("Crop Stress Analysis (Current)", color="white")
            fig.canvas.draw_idle()
            return

        if label in ["Forecast_7D", "Forecast_14D"]:
            d = datasets[label]
            d_masked = np.ma.masked_invalid(d)
            img_plot = ax_map.imshow(d_masked, cmap="RdYlGn_r", vmin=0.0, vmax=1.0)
            cbar = plt.colorbar(img_plot, ax=ax_map, fraction=0.03)
            ax_map.set_title(f"Crop Stress Forecast ({label})", color="white")
            fig.canvas.draw_idle()
            return

        d = datasets[label]
        d_masked = np.ma.masked_invalid(d)

        if label == "NDVI":
            cmap, vmin, vmax = "RdYlGn", 0.1, 0.9
        elif label == "NDRE":
            cmap, vmin, vmax = "RdYlGn", 0.1, 0.6
        elif label == "NDMI":
            cmap, vmin, vmax = "RdBu", -0.2, 0.4
        elif label == "NDWI":
            cmap, vmin, vmax = "Blues", -0.5, 0.5
        else:
            cmap, vmin, vmax = "viridis", None, None

        img_plot = ax_map.imshow(d_masked, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(img_plot, ax=ax_map, fraction=0.03)
        ax_map.set_title(f"{label} Index", color="white")
        fig.canvas.draw_idle()

    def on_hover(event):
        if event.inaxes != ax_map or event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)

        if (x < 0 or y < 0 or
            y >= datasets["NDVI"].shape[0] or x >= datasets["NDVI"].shape[1]):
            return

        v_cls = datasets["Classification"][y, x]
        v_ndvi = datasets["NDVI"][y, x]
        v_ndmi = datasets["NDMI"][y, x]
        v_ndwi = datasets["NDWI"][y, x]
        v_score = datasets["Stress_Score"][y, x]

        coord_text.set_text(f"Pixel: ({x}, {y})")

        cls_txt = "Water" if v_cls == 1 else "Soil" if v_cls == 2 else "Vegetation" if v_cls == 3 else "Cloud/Snow"
        stat_texts[0].set_text(f"Class: {cls_txt}")
        stat_texts[0].set_color("#00b894" if v_cls == 3 else "#0984e3" if v_cls == 1 else "#b2bec3")

        stat_texts[1].set_text(f"NDVI: {v_ndvi:.2f}" if np.isfinite(v_ndvi) else "NDVI: N/A")
        stat_texts[2].set_text(f"NDMI: {v_ndmi:.2f}" if np.isfinite(v_ndmi) else "NDMI: N/A")
        stat_texts[3].set_text(f"NDWI: {v_ndwi:.2f}" if np.isfinite(v_ndwi) else "NDWI: N/A")

        if v_cls == 3 and np.isfinite(v_score):
            stat_texts[4].set_text(f"Stress: {int(v_score * 100)}%")

            if v_score < 0.3:
                interp = "HEALTHY\n- Good moisture\n- Good chlorophyll"
                c = "#00b894"
            elif v_score < 0.6:
                interp = "WARNING\n- Moisture dropping\n- Check irrigation"
                c = "#fdcb6e"
            else:
                interp = "CRITICAL\n- Low moisture / chlorophyll\n- Irrigation needed"
                c = "#d63031"

            desc_text.set_text(interp)
            stat_texts[4].set_color(c)
        else:
            stat_texts[4].set_text("Stress: N/A")
            desc_text.set_text("")
            stat_texts[4].set_color("white")

        fig.canvas.draw_idle()

    radio.on_clicked(update_map)
    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    update_map("Stress_Analysis")
    plt.show()


if __name__ == "__main__":
    app = MapInputGUI()
    params = app.run()
    if params:
        try:
            res = process_sentinel2_data(params['bbox'], params['date_range'], params['max_cloud'])
            show_results(res, bbox=params["bbox"])
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Processing Error", str(e))
