# app.py
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import matplotlib.cm as cm
from PIL import Image
from agri_timeseries import compute_weekly_series
from agri_core import (
    process_sentinel2_data,
    load_datasets_from_files,
    _normalize_bbox_wsen,
    _get_forecast_for_panel_from_arrays,
    _compute_aoi_current_stress,
    _compute_quality_metrics,
    _choose_best_cultivation_week,

    # NEW ANALYTICS
    detect_stress_hotspots,
    compute_stress_trend
)

from forecast_predictor import cultivation_recommendation


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Agri-AI Web", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


# -----------------------------
# Schemas
# -----------------------------
class AnalyzeRequest(BaseModel):
    bbox: List[float] = Field(..., description="[west,south,east,north] EPSG:4326")
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    max_cloud: float = Field(20, ge=0, le=100)


class AnalyzeResponse(BaseModel):
    request_id: str
    bbox: List[float]
    date_range: str
    max_cloud: float
    selected_outputs: Dict[str, str]
    quicklooks: Dict[str, str]
    raster_size: Dict[str, int]

    current_stress: Optional[float] = None
    forecast: Optional[Dict[str, float]] = None
    advisory: Optional[str] = None

    # NEW
    quality: Optional[Dict[str, float]] = None
    cultivation_decision: Optional[Dict[str, str]] = None
    stress_trend: Optional[Dict[str, Any]] = None
    hotspots: Optional[List[Dict[str, float]]] = None


class PixelInfoRequest(BaseModel):
    request_id: str
    files: Dict[str, str]
    x: Optional[int] = None
    y: Optional[int] = None
    nx: Optional[float] = None
    ny: Optional[float] = None


class PixelExplain(BaseModel):
    status: str
    why: str
    solution: str


class PixelInfoResponse(BaseModel):
    x: int
    y: int
    class_id: int
    class_name: str
    ndvi: Optional[float] = None
    ndmi: Optional[float] = None
    ndre: Optional[float] = None
    stress: Optional[float] = None
    explain: PixelExplain


# -----------------------------
# Helpers
# -----------------------------
def _utc_today_range(days_back: int = 20):
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days_back)
    return start.isoformat(), end.isoformat()


def _isfinite(v) -> bool:
    return v is not None and float(v) == float(v) and np.isfinite(float(v))


def _class_name(c: int) -> str:
    return {
        1: "Water",
        2: "Soil",
        3: "Vegetation",
        4: "Cloud/Shadow",
    }.get(int(c), "Unknown")


def _build_specific_advisory(curr, y7, y14, quality, decision):
    if curr is None or not np.isfinite(curr):
        return (
            "Field condition: unavailable. "
            "The system could not extract a reliable current stress value for this area."
        )

    veg_ratio = float((quality or {}).get("veg_ratio", 0.0))
    valid_ratio = float((quality or {}).get("valid_stress_ratio", 0.0))
    cloud_ratio = float((quality or {}).get("cloud_ratio", 1.0))

    # 1) Current condition
    if curr >= 0.75:
        condition = "Field condition: high stress."
        meaning = "The crop may be under significant pressure."
        action = "What to do now: inspect irrigation, check for visible plant damage, and review the most stressed zones as soon as possible."
    elif curr >= 0.55:
        condition = "Field condition: moderate stress."
        meaning = "The crop shows visible pressure, but the situation is not yet extreme."
        action = "What to do now: inspect the field soon, especially the yellow and red areas, and verify irrigation uniformity."
    elif curr >= 0.35:
        condition = "Field condition: mild stress."
        meaning = "The crop is not in a critical state, but some pressure is present."
        action = "What to do now: continue monitoring and compare the next image before taking strong action."
    else:
        condition = "Field condition: low stress."
        meaning = "The crop currently appears stable and relatively healthy."
        action = "What to do now: maintain normal monitoring and no urgent intervention is needed."

    # 2) Short-term trend
    if np.isfinite(y7) and np.isfinite(y14):
        if y7 > curr + 0.05 and y14 >= y7:
            trend = "Outlook: stress is likely to increase over the next two weeks."
        elif y7 < curr - 0.05 and y14 <= y7:
            trend = "Outlook: stress is likely to decrease over the next two weeks."
        elif abs(y7 - curr) <= 0.05 and abs(y14 - curr) <= 0.05:
            trend = "Outlook: stress is expected to remain relatively stable in the short term."
        else:
            trend = "Outlook: slight changes are expected in the coming days, but no major shift is detected."
    else:
        trend = "Outlook: short-term forecast is not fully available."

    # 3) Quality note
    if cloud_ratio > 0.35:
        quality_note = "Confidence note: cloud cover is high, so this recommendation should be interpreted cautiously."
    elif valid_ratio < 0.50:
        quality_note = "Confidence note: only part of the vegetation has valid stress values, so confidence is limited."
    elif veg_ratio < 0.20:
        quality_note = "Confidence note: vegetation coverage inside the selected area is low, so the result may mix crop and non-crop surfaces."
    else:
        quality_note = "Confidence note: data quality is acceptable for field interpretation."

    # 4) Best timing
    if decision and decision.get("best_week"):
        timing = (
            f"Best timing: {decision.get('best_week')}, "
            f"with {decision.get('confidence', 'unknown').lower()} confidence."
        )
    else:
        timing = "Best timing: no clear cultivation window could be determined."

    return f"{condition} {meaning} {trend} {action} {quality_note} {timing}"


# -----------------------------
# QUICKLOOK HELPERS 
# -----------------------------


def _save_png(path_png: str, arr: np.ndarray, mode: str):
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    a = np.asarray(arr)

    def _upsample_for_display(img: Image.Image) -> Image.Image:
        min_display_width = 1400
        if img.width < min_display_width:
            scale = max(1, int(np.ceil(min_display_width / img.width)))
            img = img.resize(
                (img.width * scale, img.height * scale),
                Image.Resampling.BICUBIC
            )
        return img

    # RGB
    if mode == "rgb":
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        img = Image.fromarray(a, mode="RGB")
        img = _upsample_for_display(img)
        img.save(path_png)
        return

    # RGBA
    if mode == "rgba":
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        img = Image.fromarray(a, mode="RGBA")
        img = _upsample_for_display(img)
        img.save(path_png)
        return

    # CLASS MAP
    if mode == "class":
        a = a.astype(np.uint8)
        h, w = a.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        rgba[a == 1] = [51, 153, 255, 255]
        rgba[a == 2] = [153, 128, 102, 255]
        rgba[a == 3] = [51, 204, 102, 255]
        rgba[a == 4] = [240, 240, 240, 255]

        img = Image.fromarray(rgba, mode="RGBA")
        img = _upsample_for_display(img)
        img.save(path_png)
        return

    # Scalar maps
    a = a.astype(np.float32)
    mask = ~np.isfinite(a)

    if mode in ("ndvi", "ndre"):
        cmap, vmin, vmax = cm.get_cmap("RdYlGn"), 0.0, 1.0
    elif mode == "ndmi":
        cmap, vmin, vmax = cm.get_cmap("RdBu"), -0.5, 0.5
    elif mode == "ndwi":
        cmap, vmin, vmax = cm.get_cmap("Blues"), -0.5, 0.5
    elif mode == "stress":
        cmap, vmin, vmax = cm.get_cmap("RdYlGn_r"), 0.0, 1.0
    else:
        cmap = cm.get_cmap("viridis")
        vmin = float(np.nanmin(a))
        vmax = float(np.nanmax(a))

    x = (a - vmin) / (vmax - vmin + 1e-8)
    x = np.clip(x, 0.0, 1.0)

    rgba = (cmap(x) * 255).astype(np.uint8)
    rgba[mask] = [0, 0, 0, 0]

    img = Image.fromarray(rgba, mode="RGBA")
    img = _upsample_for_display(img)
    img.save(path_png)


def _build_stress_analysis_rgba(datasets: Dict[str, Any]) -> np.ndarray:
    cls = datasets["Classification"].astype(np.uint8)
    score = datasets["Stress_Score"].astype(np.float32)

    h, w = cls.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    rgba[cls == 1] = [51, 153, 255, 255]
    rgba[cls == 2] = [153, 128, 102, 255]
    rgba[cls == 4] = [240, 240, 240, 255]

    veg = (cls == 3) & np.isfinite(score)

    if np.any(veg):
        cmap = cm.get_cmap("RdYlGn_r")
        colored = (cmap(np.clip(score, 0.0, 1.0)) * 255).astype(np.uint8)
        rgba[veg] = colored[veg]

    return rgba


def _make_quicklooks(request_id: str, datasets: Dict[str, Any]) -> Dict[str, str]:

    out_dir = os.path.join(OUTPUTS_DIR, request_id)
    os.makedirs(out_dir, exist_ok=True)

    quicklooks = {}

    if "True_Color" in datasets:
        p = os.path.join(out_dir, "True_Color.png")
        _save_png(p, datasets["True_Color"], "rgb")
        quicklooks["True_Color"] = f"/outputs/{request_id}/True_Color.png"

    if "Classification" in datasets and "Stress_Score" in datasets:
        rgba = _build_stress_analysis_rgba(datasets)
        p = os.path.join(out_dir, "Stress_Analysis.png")
        _save_png(p, rgba, "rgba")
        quicklooks["Stress_Analysis"] = f"/outputs/{request_id}/Stress_Analysis.png"

    for key, mode in [
        ("Stress_Score", "stress"),
        ("NDVI", "ndvi"),
        ("NDMI", "ndmi"),
        ("NDRE", "ndre"),
        ("NDWI", "ndwi"),
        ("Classification", "class"),
    ]:

        if key in datasets:
            p = os.path.join(out_dir, f"{key}.png")
            _save_png(p, datasets[key], mode)
            quicklooks[key] = f"/outputs/{request_id}/{key}.png"

    return quicklooks



# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/favicon.ico")
def favicon():
    return Response(content=b"", media_type="image/x-icon")


@app.get("/", response_class=HTMLResponse)
def home():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/default-dates")
def default_dates():
    start, end = _utc_today_range(days_back=20)
    return {"start_date": start, "end_date": end}


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):

    request_id = uuid.uuid4().hex[:12]
    bbox = _normalize_bbox_wsen(req.bbox)
    date_range = f"{req.start_date}/{req.end_date}"

    files = process_sentinel2_data(bbox, date_range, req.max_cloud)
    datasets = load_datasets_from_files(files)

    # --------------------------------------------------
    # ADVANCED ANALYTICS
    # --------------------------------------------------

    # Stress hotspots
    hotspots = detect_stress_hotspots(datasets)

    # Trend (for now single value placeholder)
    stress_series = []

    if "Stress_Score" in datasets:
        arr = datasets["Stress_Score"]
        stress_series.append(float(np.nanmean(arr[np.isfinite(arr)])))

    trend = compute_stress_trend(stress_series) if stress_series else None

    ref = datasets.get("NDVI")
    if ref is None:
        return JSONResponse({"error": "Missing NDVI output"}, status_code=500)

    h, w = ref.shape

    # ---- Forecast + Current Stress ----
    forecast = _get_forecast_for_panel_from_arrays(bbox, datasets)
    curr = _compute_aoi_current_stress(datasets)

    # ---- Advisory ----
    advisory = None
    y7 = np.nan
    y14 = np.nan

    if isinstance(forecast, dict):
        y7 = float(forecast.get("y_7", np.nan))
        y14 = float(forecast.get("y_14", np.nan))

    quality = _compute_quality_metrics(datasets)

    decision = None
    if isinstance(forecast, dict):
        decision = _choose_best_cultivation_week(
            curr,
            forecast.get("y_7"),
            forecast.get("y_14"),
            quality
        )

    advisory = _build_specific_advisory(curr, y7, y14, quality, decision)

    # ---- NEW: AOI QUALITY ----
    quality = _compute_quality_metrics(datasets)

    # ---- NEW: CULTIVATION DECISION ----
    decision = None
    if isinstance(forecast, dict):
        decision = _choose_best_cultivation_week(
            curr,
            forecast.get("y_7"),
            forecast.get("y_14"),
            quality
        )

    quicklooks = _make_quicklooks(request_id, datasets)

    return AnalyzeResponse(
        request_id=request_id,
        bbox=bbox,
        date_range=date_range,
        max_cloud=req.max_cloud,
        selected_outputs=files,
        quicklooks=quicklooks,
        raster_size={"width": int(w), "height": int(h)},

        current_stress=float(curr) if np.isfinite(curr) else None,
        forecast=forecast if isinstance(forecast, dict) else None,
        advisory=advisory,

        quality=quality,
        cultivation_decision=decision,

        # NEW ANALYTICS
        stress_trend=trend,
        hotspots=hotspots
    )


@app.post("/api/analyze-timeseries")
def analyze_timeseries(req: AnalyzeRequest):
    bbox = _normalize_bbox_wsen(req.bbox)
    date_range = f"{req.start_date}/{req.end_date}"

    images, series = compute_weekly_series(
        bbox,
        date_range,
        req.max_cloud
    )

    enriched_images = []

    for i, item in enumerate(images):
        files = item.get("files", {})
        date = item.get("date")
        stress = item.get("stress")

        try:
            datasets = load_datasets_from_files(files)
            ts_request_id = f"ts_{uuid.uuid4().hex[:10]}_{i}"
            quicklooks = _make_quicklooks(ts_request_id, datasets)

            enriched_images.append({
                "date": date,
                "stress": stress,
                "quicklook": (
                    quicklooks.get("Stress_Analysis")
                    or quicklooks.get("True_Color")
                    or quicklooks.get("Stress_Score")
                )
            })
        except Exception as e:
            print("timeline quicklook skipped:", e)
            enriched_images.append({
                "date": date,
                "stress": stress,
                "quicklook": None
            })

    stress_values = [x["stress"] for x in series if x["stress"] is not None]
    trend = compute_stress_trend(stress_values) if stress_values else None

    return {
        "images": enriched_images,
        "stress_series": series,
        "trend": trend
    }


@app.post("/api/pixel-info", response_model=PixelInfoResponse)
def pixel_info(req: PixelInfoRequest):

    datasets = load_datasets_from_files(req.files)

    ref = datasets.get("NDVI")
    if ref is None:
        return JSONResponse({"error": "Missing NDVI raster in files"}, status_code=400)

    h, w = ref.shape

    if req.nx is not None and req.ny is not None:
        nx = min(max(float(req.nx), 0.0), 1.0)
        ny = min(max(float(req.ny), 0.0), 1.0)
        x = int(round(nx * (w - 1)))
        y = int(round(ny * (h - 1)))
    else:
        x = int(req.x) if req.x is not None else 0
        y = int(req.y) if req.y is not None else 0

    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))

    cls = datasets.get("Classification", None)
    v_cls = int(cls[y, x]) if cls is not None else 0

    ndvi = datasets.get("NDVI", None)
    ndmi = datasets.get("NDMI", None)
    ndre = datasets.get("NDRE", None)
    stress = datasets.get("Stress_Score", None)

    v_ndvi = float(ndvi[y, x]) if ndvi is not None and np.isfinite(ndvi[y, x]) else None
    v_ndmi = float(ndmi[y, x]) if ndmi is not None and np.isfinite(ndmi[y, x]) else None
    v_ndre = float(ndre[y, x]) if ndre is not None and np.isfinite(ndre[y, x]) else None
    v_stress = float(stress[y, x]) if stress is not None and np.isfinite(stress[y, x]) else None

    from agri_core import compute_pixel_explanation

    explanation = compute_pixel_explanation(datasets, x, y)

    return PixelInfoResponse(
    x=x,
    y=y,
    class_id=v_cls,
    class_name=_class_name(v_cls),
    ndvi=v_ndvi,
    ndmi=v_ndmi,
    ndre=v_ndre,
    stress=v_stress,
    explain=PixelExplain(
        status=explanation.get("status", ""),
        why=explanation.get("why", ""),
        solution=explanation.get("solution", "")
    )
)


@app.get("/api/models")
def list_models():
    if not os.path.exists(MODELS_DIR):
        return {"models": []}

    models = []
    for name in os.listdir(MODELS_DIR):
        if name.lower().endswith((".joblib", ".pkl")):
            models.append({"name": name, "url": f"/api/models/{name}"})
    return {"models": models}


@app.get("/api/models/{model_name}")
def download_model(model_name: str):
    path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(path):
        return JSONResponse({"error": "Model not found"}, status_code=404)
    return FileResponse(path, filename=model_name)