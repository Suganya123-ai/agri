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
    raster_size: Dict[str, int]              # {"width": W, "height": H}
    current_stress: Optional[float] = None
    forecast: Optional[Dict[str, float]] = None
    advisory: Optional[str] = None


class PixelInfoRequest(BaseModel):
    request_id: str
    files: Dict[str, str]
    # either x/y OR nx/ny
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


def _explain_pixel(v_cls: int, ndvi, ndmi, ndre, stress) -> PixelExplain:
    cname = _class_name(v_cls)

    if v_cls != 3:
        if v_cls == 1:
            return PixelExplain(
                status="Not a crop pixel (Water)",
                why="This pixel is water, not crops.",
                solution="Draw the rectangle over cultivated fields (vegetation).",
            )
        if v_cls == 2:
            return PixelExplain(
                status="Not a crop pixel (Soil)",
                why="This pixel is mostly bare soil (low vegetation cover).",
                solution="Select an area with crops (greener area / higher NDVI).",
            )
        if v_cls == 4:
            return PixelExplain(
                status="Not reliable (Cloud/Shadow)",
                why="Cloud/shadow contamination makes indices unreliable.",
                solution="Try another date range or increase max cloud slightly.",
            )
        return PixelExplain(
            status="Unknown area",
            why="This pixel is not confidently vegetation.",
            solution="Try another area / zoom in / redraw the AOI.",
        )

    if not _isfinite(stress):
        return PixelExplain(
            status="Vegetation but missing stress",
            why="Stress is masked/invalid for this pixel (often cloud mask edges).",
            solution="Click nearby vegetation pixel or try different dates.",
        )

    s = float(stress)
    moisture_low = _isfinite(ndmi) and float(ndmi) < 0.05
    chlor_low = _isfinite(ndre) and float(ndre) < 0.20

    drivers = []
    if moisture_low:
        drivers.append("low moisture")
    if chlor_low:
        drivers.append("low chlorophyll")
    if not drivers:
        drivers.append("heat/irrigation timing/other stress factors")

    drivers_text = ", ".join(drivers)

    if s < 0.30:
        return PixelExplain(
            status="Healthy ✅",
            why=f"Stress is low. Main signals: {drivers_text}.",
            solution="Keep current irrigation schedule and monitor regularly.",
        )
    elif s < 0.60:
        return PixelExplain(
            status="Moderate stress ⚠️",
            why=f"Stress is increasing. Likely: {drivers_text}.",
            solution="Check irrigation timing, soil moisture, and inspect plants.",
        )
    else:
        return PixelExplain(
            status="High stress 🛑",
            why=f"Stress is high. Likely: {drivers_text}.",
            solution="Prioritize irrigation, inspect for pests/disease, consider protective measures.",
        )


def _save_png(path_png: str, arr: np.ndarray, mode: str):
    """
    Save quicklooks at native pixel resolution (NO matplotlib resize).
    Supports: rgb, class, ndvi/ndmi/ndre/ndwi/stress (scalar), rgba (already colored).
    """
    import os
    import numpy as np
    from PIL import Image
    import matplotlib.cm as cm

    os.makedirs(os.path.dirname(path_png), exist_ok=True)

    a = np.asarray(arr)

    # --------- NEW: direct RGBA saving ----------
    if mode == "rgba":
        # Fix common bad shapes like (1,H,W,4) or (H,W,4,1)
        if a.ndim == 4 and a.shape[0] == 1:
            a = a[0]                 # (H,W,4)
        if a.ndim == 4 and a.shape[-1] == 1:
            a = a[..., 0]            # (H,W,4)
        if a.ndim != 3 or a.shape[-1] != 4:
            raise ValueError(f"RGBA image must be (H,W,4). Got {a.shape}")

        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)

        Image.fromarray(a, mode="RGBA").save(path_png, format="PNG", optimize=True)
        return

    # --------- RGB ----------
    if mode == "rgb":
        if a.ndim == 4 and a.shape[0] == 1:
            a = a[0]
        if a.ndim != 3 or a.shape[-1] != 3:
            raise ValueError(f"RGB image must be (H,W,3). Got {a.shape}")

        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)

        Image.fromarray(a, mode="RGB").save(path_png, format="PNG", optimize=True)
        return

    # --------- CLASS ----------
    if mode == "class":
        a = a.astype(np.uint8)
        if a.ndim != 2:
            if a.ndim == 3 and a.shape[0] == 1:
                a = a[0]
            else:
                raise ValueError(f"Class map must be (H,W). Got {a.shape}")

        h, w = a.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[a == 1] = [51, 153, 255, 255]    # Water
        rgba[a == 2] = [153, 128, 102, 255]   # Soil
        rgba[a == 3] = [51, 204, 102, 255]    # Veg
        rgba[a == 4] = [245, 245, 245, 255]   # Cloud/Shadow

        Image.fromarray(rgba, mode="RGBA").save(path_png, format="PNG", optimize=True)
        return

    # --------- SCALAR maps ----------
    # Must be 2D
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"Scalar map must be (H,W). Got {a.shape}")

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
        vmin = float(np.nanmin(a)) if np.isfinite(a).any() else 0.0
        vmax = float(np.nanmax(a)) if np.isfinite(a).any() else 1.0

    den = (vmax - vmin) if vmax != vmin else 1.0
    x = (a - vmin) / den
    x = np.clip(x, 0.0, 1.0)

    rgba = (cmap(x) * 255).astype(np.uint8)
    rgba[mask] = [0, 0, 0, 0]  # transparent where NaN

    Image.fromarray(rgba, mode="RGBA").save(path_png, format="PNG", optimize=True)



def _build_stress_analysis_rgba(datasets: Dict[str, Any]) -> np.ndarray:
    """
    Returns a desktop-like composite RGBA image:
    - Water / Soil / Cloud as fixed colors
    - Vegetation colored by Stress_Score using RdYlGn_r
    Output: uint8 array (H, W, 4)
    """
    import numpy as np
    import matplotlib.cm as cm

    cls = np.asarray(datasets["Classification"])
    score = np.asarray(datasets["Stress_Score"], dtype=np.float32)

    # Make sure cls is 2D
    if cls.ndim == 3 and cls.shape[0] == 1:
        cls = cls[0]
    if cls.ndim != 2:
        raise ValueError(f"Classification must be (H,W). Got {cls.shape}")

    # Make sure score is 2D
    if score.ndim == 3 and score.shape[0] == 1:
        score = score[0]
    if score.ndim != 2:
        raise ValueError(f"Stress_Score must be (H,W). Got {score.shape}")

    h, w = cls.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Fixed colors
    rgba[cls == 1] = [51, 153, 255, 255]     # Water
    rgba[cls == 2] = [153, 128, 102, 255]    # Soil
    rgba[cls == 4] = [245, 245, 245, 255]    # Cloud / Shadow

    # Vegetation colored by stress
    veg = (cls == 3) & np.isfinite(score)
    if np.any(veg):
        cmap = cm.get_cmap("RdYlGn_r")
        colors = (cmap(np.clip(score[veg], 0.0, 1.0)) * 255).astype(np.uint8)
        rgba[veg] = colors  # already RGBA

    return rgba


def _make_quicklooks(request_id: str, datasets: Dict[str, Any]) -> Dict[str, str]:
    out_dir = os.path.join(OUTPUTS_DIR, request_id)
    os.makedirs(out_dir, exist_ok=True)

    quicklooks = {}

    # True color
    if "True_Color" in datasets:
        p = os.path.join(out_dir, "True_Color.png")
        _save_png(p, datasets["True_Color"], mode="rgb")
        quicklooks["True_Color"] = f"/outputs/{request_id}/True_Color.png"

    # Desktop-like composite (Stress_Analysis)
    if "Classification" in datasets and "Stress_Score" in datasets:
        rgba = _build_stress_analysis_rgba(datasets)  # MUST be (H,W,4) uint8

        # ✅ safety squeeze (fixes (1,H,W,4) etc)
        rgba = np.asarray(rgba)
        if rgba.ndim == 4 and rgba.shape[0] == 1:
            rgba = rgba[0]
        if rgba.ndim != 3 or rgba.shape[-1] != 4:
            raise ValueError(f"_build_stress_analysis_rgba returned invalid shape: {rgba.shape}")

        p = os.path.join(out_dir, "Stress_Analysis.png")
        _save_png(p, rgba, mode="rgba")
        quicklooks["Stress_Analysis"] = f"/outputs/{request_id}/Stress_Analysis.png"

    # Maps
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



def _build_stress_analysis_rgba(datasets: Dict[str, Any]) -> np.ndarray:
    """
    Desktop-like composite:
    - Water/Soil/Cloud as fixed colors
    - Vegetation colored by stress colormap
    """
    import matplotlib.cm as cm

    cls = datasets["Classification"].astype(np.uint8)
    score = datasets["Stress_Score"].astype(np.float32)

    h, w = cls.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # base classes
    rgba[cls == 1] = [ 51, 153, 255, 255]  # Water
    rgba[cls == 2] = [153, 128, 102, 255]  # Soil
    rgba[cls == 4] = [240, 240, 240, 255]  # Cloud/Shadow

    veg_mask = (cls == 3) & np.isfinite(score)
    if np.any(veg_mask):
        cmap = cm.get_cmap("RdYlGn_r")  # green->low stress, red->high
        x = np.clip(score, 0.0, 1.0)
        colored = (cmap(x) * 255).astype(np.uint8)
        rgba[veg_mask] = colored[veg_mask]

    # non-veg pixels that are 0 => transparent
    rgba[(cls == 0)] = [0, 0, 0, 0]
    return rgba

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

    # raster size for frontend mapping sanity
    ref = datasets.get("NDVI")
    if ref is None:
        return JSONResponse({"error": "Missing NDVI output"}, status_code=500)
    h, w = ref.shape

    forecast = _get_forecast_for_panel_from_arrays(bbox, datasets)
    curr = _compute_aoi_current_stress(datasets)

    advisory = None
    if isinstance(forecast, dict) and np.isfinite(curr):
        y7 = float(forecast.get("y_7", np.nan))
        y14 = float(forecast.get("y_14", np.nan))
        if np.isfinite(y7) and np.isfinite(y14):
            advisory = cultivation_recommendation(float(curr), y7, y14)

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
    )


@app.post("/api/pixel-info", response_model=PixelInfoResponse)
def pixel_info(req: PixelInfoRequest):
    datasets = load_datasets_from_files(req.files)

    ref = datasets.get("NDVI", None)
    if ref is None:
        return JSONResponse({"error": "Missing NDVI raster in files"}, status_code=400)

    h, w = ref.shape

    # If nx/ny provided, compute pixel safely
    if req.nx is not None and req.ny is not None:
        nx = float(req.nx)
        ny = float(req.ny)
        x = int(nx * (w - 1))
        y = int(ny * (h - 1))
    else:
        x, y = int(req.x), int(req.y)

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