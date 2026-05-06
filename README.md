# Agri-AI Web

FastAPI web app for basic field intelligence using Sentinel-2 imagery:
- Draw an AOI on the map
- Run analysis to generate indices (NDVI/NDMI/NDRE/NDWI), stress score, and classification
- View quicklooks and use the Pixel Inspector by clicking the quicklook image

## Prerequisites

- Python 3.10+ (tested with Python 3.12)
- Internet access (the app queries Microsoft Planetary Computer / Sentinel-2)

## Setup

From the project folder:

1) Create & activate a virtual environment

- macOS / Linux:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`

2) Install dependencies

- `pip install -r requirements.txt`

## Run

Start the server with Uvicorn:

- `uvicorn app:app --host 0.0.0.0 --port 8000`

Open:
- App UI: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`

Tip: for local development auto-reload:
- `uvicorn app:app --reload --host 0.0.0.0 --port 8000`

## How to use

1) Open the UI (`/`).
2) Use the rectangle draw tool on the map to select your field (AOI).
3) Choose date range and max cloud cover.
4) Click **Start Analysis**.
5) Inspect the results:
   - KPIs (Current Stress + forecast)
   - Quicklook raster previews (True Color / Stress / NDVI / etc.)
   - **Pixel Inspector**: click directly on the quicklook image to populate pixel values (NDVI/NDMI/NDRE/stress) and the explanation.

## Outputs

The app generates:
- Quicklook PNGs under `outputs/<request_id>/` (served at `/outputs/...`).
- GeoTIFF layers are written by default to: `~/Downloads/Agri_AI_Results/`

## Notes

- The map UI uses locally vendored Leaflet assets under `static/vendor/` (so it works even when public CDNs are blocked).
- If you are running over a remote connection (VS Code Remote/SSH), forward port `8000` and open the forwarded URL.

## Troubleshooting

- **Port already in use**: change the port, e.g. `--port 8001`.
- **Pixel Inspector shows `--`**: ensure analysis completed successfully and click on the quicklook image (the inspector uses the analysis output rasters).
- **rasterio install issues (macOS)**: try upgrading pip first: `pip install -U pip`, then reinstall requirements.
