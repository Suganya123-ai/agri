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

from agri_core import (
    process_sentinel2_data,
    load_datasets_from_files,
    _normalize_bbox_wsen,
    _get_forecast_for_panel_from_arrays,
    _compute_aoi_current_stress,
)

from forecast_predictor import cultivation_recommendation


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")  # quicklook PNGs
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUTS_DIR, exist_ok=True)


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Agri-AI Web", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
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
    selected_outputs: Dict[str, str]          # GeoTIFF paths (server-side)
    quicklooks: Dict[str, str]               # URLs for browser
    current_stress: Optional[float] = None
    forecast: Optional[Dict[str, float]] = None
    advisory: Optional[str] = None

    # ✅ NEW: cultivation timing fields (these feed your UI)
    best_cultivation_week: Optional[str] = None
    cultivation_confidence: Optional[str] = None
    cultivation_reason: Optional[str] = None


class PixelInfoRequest(BaseModel):
    request_id: str
    x: int
    y: int
    files: Dict[str, str]  # selected_outputs from analyze response


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


def _explain_pixel(v_cls: int, ndvi, ndmi, ndre, stress) -> PixelExplain:
    cname = _class_name(v_cls)

    if v_cls != 3:
        if v_cls == 1:
            return PixelExplain(
                status="Not a crop pixel (Water)",
                why="This pixel is classified as water, not a cultivated plant area.",
                solution="Select a rectangle that covers farmland (vegetation)."
            )
        if v_cls == 2:
            return PixelExplain(
                status="Not a crop pixel (Soil)",
                why="This pixel is mostly bare soil (low vegetation cover).",
                solution="Select an area with visible crops (higher NDVI / green cover)."
            )
        if v_cls == 4:
            return PixelExplain(
                status="Not reliable (Cloud/Shadow)",
                why="This pixel is affected by clouds or shadow, so indices are unreliable.",
                solution="Try another date window or increase max cloud cover slightly."
            )
        return PixelExplain(
            status="Unknown area",
            why="This pixel is not confidently classified as vegetation.",
            solution="Try selecting a different area or zoom in for a better field selection."
        )

    if not _isfinite(stress):
        return PixelExplain(
            status="Vegetation, but missing stress",
            why="This pixel is vegetation, but stress could not be computed (masked or invalid).",
            solution="Try a nearby pixel or adjust date/cloud cover."
        )

    s = float(stress)
    moisture_low = _isfinite(ndmi) and float(ndmi) < 0.05
    chlor_low = _isfinite(ndre) and float(ndre) < 0.20

    drivers = []
    if moisture_low:
        drivers.append("low moisture (dryness)")
    if chlor_low:
        drivers.append("low chlorophyll (weak vegetation health)")
    if not drivers:
        drivers.append("mixed factors (heat, irrigation timing, or plant stress)")

    drivers_text = ", ".join(drivers)

    if s < 0.30:
        return PixelExplain(
            status="Healthy ✅",
            why=f"Stress is low. Conditions look good. Main indicators: {drivers_text}.",
            solution="Keep current irrigation schedule and continue monitoring."
        )
    elif s < 0.60:
        return PixelExplain(
            status="Moderate stress ⚠️",
            why=f"Stress is rising. Likely reasons: {drivers_text}.",
            solution="Check irrigation timing, soil moisture, and inspect plants for early stress signs."
        )
    else:
        return PixelExplain(
            status="High stress 🛑",
            why=f"Stress is high. Likely reasons: {drivers_text}.",
            solution="Prioritize irrigation, inspect for disease/pests, and consider protective measures."
        )


def _save_png(path_png: str, arr: np.ndarray, mode: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path_png), exist_ok=True)

    plt.figure(figsize=(8, 8), dpi=140)
    plt.axis("off")

    if mode == "rgb":
        plt.imshow(arr)
    elif mode == "class":
        h, w = arr.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[arr == 1] = [0.2, 0.6, 1.0, 1.0]     # Water
        rgba[arr == 2] = [0.6, 0.5, 0.4, 1.0]     # Soil
        rgba[arr == 3] = [0.2, 0.8, 0.3, 1.0]     # Veg
        rgba[arr == 4] = [0.95, 0.95, 0.95, 1.0]  # Cloud/Shadow
        plt.imshow(rgba)
    else:
        a = np.array(arr, dtype=np.float32)
        a = np.ma.masked_invalid(a)

        if mode in ("ndvi", "ndre"):
            cmap, vmin, vmax = "RdYlGn", 0.0, 1.0
        elif mode == "ndmi":
            cmap, vmin, vmax = "RdBu", -0.5, 0.5
        elif mode == "ndwi":
            cmap, vmin, vmax = "Blues", -0.5, 0.5
        elif mode == "stress":
            cmap, vmin, vmax = "RdYlGn_r", 0.0, 1.0
        else:
            cmap, vmin, vmax = "viridis", None, None

        plt.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax)

    plt.tight_layout(pad=0)
    plt.savefig(path_png, bbox_inches="tight", pad_inches=0)
    plt.close()


def _make_quicklooks(request_id: str, datasets: Dict[str, Any]) -> Dict[str, str]:
    out_dir = os.path.join(OUTPUTS_DIR, request_id)
    os.makedirs(out_dir, exist_ok=True)

    quicklooks = {}

    if "True_Color" in datasets:
        p = os.path.join(out_dir, "True_Color.png")
        _save_png(p, datasets["True_Color"], mode="rgb")
        quicklooks["True_Color"] = f"/outputs/{request_id}/True_Color.png"

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
            _save_png(p, datasets[key], mode=mode)
            quicklooks[key] = f"/outputs/{request_id}/{key}.png"

    return quicklooks


# ✅ NEW: cultivation week picker (simple & understandable)
def _choose_best_cultivation_week(curr: float, y7: float, y14: float) -> Dict[str, str]:
    """
    Human-friendly rule:
    - We prefer the week with LOWER predicted stress.
    - If both are low -> "This week" is fine.
    """
    # Safety: handle NaNs
    if not (np.isfinite(curr) and np.isfinite(y7) and np.isfinite(y14)):
        return {
            "best_week": "N/A",
            "confidence": "Low",
            "reason": "Forecast values are missing or invalid, so cultivation timing cannot be estimated."
        }

    # classify stress level
    def level(v):
        if v < 0.30:
            return "low"
        if v < 0.60:
            return "medium"
        return "high"

    lcurr, l7, l14 = level(curr), level(y7), level(y14)

    # choose the minimum forecast
    best_week = "Next 7 days" if y7 <= y14 else "Next 14 days"
    best_val = min(y7, y14)

    # confidence: clearer separation -> higher confidence
    diff = abs(y7 - y14)
    if diff >= 0.08:
        conf = "High"
    elif diff >= 0.03:
        conf = "Medium"
    else:
        conf = "Low"

    reason = (
        f"We compared predicted stress for the next 7 days ({y7:.2f}) "
        f"and next 14 days ({y14:.2f}). Lower stress is better for cultivation, "
        f"so the recommendation is: {best_week}."
    )

    # add easy interpretation
    if best_val < 0.30:
        reason += " Conditions look favorable (low stress)."
    elif best_val < 0.60:
        reason += " Conditions are moderate — irrigation planning is important."
    else:
        reason += " Conditions look difficult (high stress) — consider delaying or preparing mitigation."

    return {"best_week": best_week, "confidence": conf, "reason": reason}


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

    # 1) Run satellite processing (GeoTIFFs)
    files = process_sentinel2_data(bbox, date_range, req.max_cloud)

    # 2) Load arrays
    datasets = load_datasets_from_files(files)

    # 3) Forecast + advisory + cultivation
    forecast = _get_forecast_for_panel_from_arrays(bbox, datasets)
    curr = _compute_aoi_current_stress(datasets)

    advisory = None
    best_week = None
    cult_conf = None
    cult_reason = None

    if isinstance(forecast, dict) and np.isfinite(curr):
        y7 = float(forecast.get("y_7", np.nan))
        y14 = float(forecast.get("y_14", np.nan))

        if np.isfinite(y7) and np.isfinite(y14):
            advisory = cultivation_recommendation(float(curr), y7, y14)

            pick = _choose_best_cultivation_week(float(curr), y7, y14)
            best_week = pick["best_week"]
            cult_conf = pick["confidence"]
            cult_reason = pick["reason"]

    # 4) quicklooks (PNGs)
    quicklooks = _make_quicklooks(request_id, datasets)

    return AnalyzeResponse(
        request_id=request_id,
        bbox=bbox,
        date_range=date_range,
        max_cloud=req.max_cloud,
        selected_outputs=files,
        quicklooks=quicklooks,
        current_stress=float(curr) if np.isfinite(curr) else None,
        forecast=forecast if isinstance(forecast, dict) else None,
        advisory=advisory,

        # ✅ send cultivation fields to frontend
        best_cultivation_week=best_week,
        cultivation_confidence=cult_conf,
        cultivation_reason=cult_reason,
    )


@app.post("/api/pixel-info", response_model=PixelInfoResponse)
def pixel_info(req: PixelInfoRequest):
    datasets = load_datasets_from_files(req.files)

    ref = datasets.get("NDVI", None)
    if ref is None:
        return JSONResponse({"error": "Missing NDVI raster in files"}, status_code=400)

    h, w = ref.shape
    x, y = int(req.x), int(req.y)

    if x < 0 or y < 0 or x >= w or y >= h:
        return JSONResponse(
            {"error": f"Pixel out of bounds. Image size is {w}x{h}."},
            status_code=400
        )

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

    explain = _explain_pixel(v_cls, v_ndvi, v_ndmi, v_ndre, v_stress)

    return PixelInfoResponse(
        x=x,
        y=y,
        class_id=v_cls,
        class_name=_class_name(v_cls),
        ndvi=v_ndvi,
        ndmi=v_ndmi,
        ndre=v_ndre,
        stress=v_stress,
        explain=explain,
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
