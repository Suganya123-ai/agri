let map;
let drawnItems;
let lastBBox = null;
let lastAnalyzeResponse = null;
let trendChartInstance = null;
let hotspotLayer = null;

const BANGALORE = [13.05, 77.60];

function setStatus(msg) {
  const el = document.getElementById("status");
  if (el) el.textContent = "Status: " + msg;
}

function safeText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

/* ===============================
   STRESS COLOR HELPER
================================= */
function applyStressColor(el, value) {
  if (!el) return;

  el.classList.remove("stress-low", "stress-medium", "stress-high");

  if (value == null || Number.isNaN(value)) return;

  if (value < 0.3) {
    el.classList.add("stress-low");
  } else if (value < 0.6) {
    el.classList.add("stress-medium");
  } else {
    el.classList.add("stress-high");
  }
}

/* ===============================
   CONFIDENCE COLOR HELPER
================================= */
function applyConfidenceColor(el, level) {
  if (!el) return;

  el.classList.remove("conf-high", "conf-medium", "conf-low");

  if (!level) return;

  const txt = String(level).toLowerCase();
  if (txt.includes("high")) {
    el.classList.add("conf-high");
  } else if (txt.includes("medium")) {
    el.classList.add("conf-medium");
  } else {
    el.classList.add("conf-low");
  }
}

function todayUTCISO() {
  const now = new Date();
  return now.toISOString().slice(0, 10);
}

function minusDaysISO(iso, days) {
  const dt = new Date(iso + "T00:00:00Z");
  dt.setUTCDate(dt.getUTCDate() - days);
  return dt.toISOString().slice(0, 10);
}

async function loadDefaultDates() {
  const end = todayUTCISO();
  const start = minusDaysISO(end, 20);

  const startEl = document.getElementById("startDate");
  const endEl = document.getElementById("endDate");

  if (startEl) startEl.value = start;
  if (endEl) endEl.value = end;
}

/* ===============================
   MAP
================================= */
function initMap() {
  map = L.map("map").setView(BANGALORE, 11);

  L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors",
    }
  ).addTo(map);

  drawnItems = new L.FeatureGroup();
  map.addLayer(drawnItems);

  hotspotLayer = new L.LayerGroup();
  map.addLayer(hotspotLayer);

  const drawControl = new L.Control.Draw({
    draw: {
      polygon: false,
      polyline: false,
      circle: false,
      marker: false,
      circlemarker: false,
      rectangle: { shapeOptions: { color: "#00b894", weight: 2 } }
    },
    edit: { featureGroup: drawnItems, remove: true }
  });

  map.addControl(drawControl);

  map.on(L.Draw.Event.CREATED, function (event) {
    drawnItems.clearLayers();
    if (hotspotLayer) hotspotLayer.clearLayers();

    const layer = event.layer;
    drawnItems.addLayer(layer);

    const b = layer.getBounds();
    lastBBox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
    setStatus("AOI selected ✅");
  });

  map.on(L.Draw.Event.DELETED, function () {
    lastBBox = null;
    if (hotspotLayer) hotspotLayer.clearLayers();
    setStatus("AOI cleared");
  });
}

function clearAOI() {
  drawnItems.clearLayers();
  if (hotspotLayer) hotspotLayer.clearLayers();
  lastBBox = null;
  lastAnalyzeResponse = null;
  setStatus("AOI cleared ✅");
}

/* ===============================
   UPDATE RESULTS
================================= */
function updateResults(resp) {
  lastAnalyzeResponse = resp;

  const curr = resp.current_stress;
  const f = resp.forecast || {};

  const currEl = document.getElementById("currStress");
  const y7El = document.getElementById("y7");
  const y14El = document.getElementById("y14");

  if (currEl) {
    currEl.textContent =
      curr == null ? "--" : `${curr.toFixed(4)} (${Math.round(curr * 100)}%)`;
    applyStressColor(currEl, curr);
  }

  if (y7El) {
    y7El.textContent =
      f.y_7 == null ? "--" : `${f.y_7.toFixed(4)} (${Math.round(f.y_7 * 100)}%)`;
  }

  if (y14El) {
    y14El.textContent =
      f.y_14 == null ? "--" : `${f.y_14.toFixed(4)} (${Math.round(f.y_14 * 100)}%)`;
  }

  safeText("advisory", "Advisory: " + (resp.advisory || "--"));

  if (resp.quality) {
    safeText("vegRatio", `${(resp.quality.veg_ratio * 100).toFixed(1)}%`);
    safeText("validRatio", `${(resp.quality.valid_stress_ratio * 100).toFixed(1)}%`);
    safeText("cloudRatio", `${(resp.quality.cloud_ratio * 100).toFixed(1)}%`);
  } else {
    safeText("vegRatio", "--");
    safeText("validRatio", "--");
    safeText("cloudRatio", "--");
  }

  if (resp.cultivation_decision) {
    const d = resp.cultivation_decision;

    safeText("bestWeek", d.best_week || "--");
    safeText("decisionReason", d.reason || "--");

    const confEl = document.getElementById("confidence");
    if (confEl) {
      confEl.textContent = d.confidence || "--";
      applyConfidenceColor(confEl, d.confidence);
    }
  } else {
    safeText("bestWeek", "--");
    safeText("decisionReason", "--");
    safeText("confidence", "--");
  }

  const layerSelect = document.getElementById("layerSelect");
  const img = document.getElementById("qlImage");

  function showLayer(layerKey) {
    if (!img || !resp.quicklooks) return;

    const url =
      resp.quicklooks[layerKey] ||
      resp.quicklooks["True_Color"] ||
      resp.quicklooks["Stress_Analysis"];

    if (url) {
      img.src = url + "?t=" + Date.now();
    }
  }

  if (layerSelect) {
    layerSelect.onchange = () => showLayer(layerSelect.value);

    layerSelect.value = resp.quicklooks?.["True_Color"]
      ? "True_Color"
      : (resp.quicklooks?.["Stress_Analysis"] ? "Stress_Analysis" : "True_Color");

    showLayer(layerSelect.value);
  }

  drawHotspots(resp.hotspots || []);
}

/* ===============================
   ANALYZE
================================= */
async function analyze() {
  if (!lastBBox) {
    alert("Please draw a rectangle (AOI) first.");
    return;
  }

  const startDate = document.getElementById("startDate")?.value;
  const endDate = document.getElementById("endDate")?.value;
  const maxCloud = parseFloat(document.getElementById("maxCloud")?.value || "20");

  setStatus("Processing...");
  const analyzeBtn = document.getElementById("analyzeBtn");
  if (analyzeBtn) analyzeBtn.disabled = true;

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        bbox: lastBBox,
        start_date: startDate,
        end_date: endDate,
        max_cloud: maxCloud
      })
    });

    if (!res.ok) throw new Error(await res.text());

    const data = await res.json();
    updateResults(data);
    await loadTimelineData();

    setStatus("Done ✅");
  } catch (e) {
    console.error(e);
    alert("Analyze failed: " + e.message);
    setStatus("Error ❌");
  } finally {
    if (analyzeBtn) analyzeBtn.disabled = false;
  }
}

/* ===============================
   PIXEL INSPECTOR
================================= */
function bindPixelInspector() {
  const img = document.getElementById("qlImage");
  if (!img) return;

  img.addEventListener("click", async (ev) => {
    if (!lastAnalyzeResponse) {
      alert("Run analysis first.");
      return;
    }

    const rect = img.getBoundingClientRect();
    if (!rect.width || !rect.height) {
      alert("Quicklook not ready yet — try again in a second.");
      return;
    }
    const nx = (ev.clientX - rect.left) / rect.width;
    const ny = (ev.clientY - rect.top) / rect.height;

    // Pixel inspector only needs a small subset of rasters.
    // Sending fewer files reduces payload size and avoids slow raster loads.
    const allFiles = lastAnalyzeResponse.selected_outputs || {};
    const neededKeys = ["NDVI", "NDMI", "NDRE", "Stress_Score", "Classification"];
    const files = {};
    neededKeys.forEach((k) => {
      if (allFiles[k]) files[k] = allFiles[k];
    });
    const filesToSend = files.NDVI ? files : allFiles;

    try {
      const res = await fetch("/api/pixel-info", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          request_id: lastAnalyzeResponse.request_id,
          nx,
          ny,
          files: filesToSend
        })
      });

      let data;
      try {
        data = await res.json();
      } catch {
        throw new Error(await res.text());
      }

      if (!res.ok) throw new Error(data.error || "Pixel inspector failed");

      safeText("pixXY", `${data.x}, ${data.y}`);
      safeText("pixClass", data.class_name || "--");
      safeText("pixNDVI", data.ndvi == null ? "--" : data.ndvi.toFixed(3));
      safeText("pixNDMI", data.ndmi == null ? "--" : data.ndmi.toFixed(3));
      safeText("pixNDRE", data.ndre == null ? "--" : data.ndre.toFixed(3));

      const stressEl = document.getElementById("pixStress");
      if (stressEl) {
        stressEl.classList.remove("stress-low", "stress-medium", "stress-high");
        if (data.stress == null) {
          stressEl.textContent = "--";
        } else {
          stressEl.textContent = `${data.stress.toFixed(3)} (${Math.round(data.stress * 100)}%)`;
          applyStressColor(stressEl, data.stress);
        }
      }

      safeText("pixelStatus", data.explain?.status || "--");
      safeText("pixelWhy", data.explain?.why || "--");
      safeText("pixelSolution", data.explain?.solution || "--");
    } catch (e) {
      console.error(e);
      alert("Pixel inspector failed: " + e.message);
    }
  });
}

/* ===============================
   MODELS
================================= */
async function loadModels() {
  const ul = document.getElementById("modelsList");
  if (!ul) return;

  ul.innerHTML = "";

  try {
    const res = await fetch("/api/models");
    const data = await res.json();

    if (!data.models || data.models.length === 0) {
      ul.innerHTML = "<li>No models found</li>";
      return;
    }

    data.models.forEach((m) => {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = m.url;
      a.textContent = m.name;
      a.target = "_blank";
      li.appendChild(a);
      ul.appendChild(li);
    });
  } catch (e) {
    console.error(e);
    ul.innerHTML = "<li>Error loading models</li>";
  }
}

/* ===============================
   HOTSPOTS
================================= */
function drawHotspots(hotspots) {
  if (!hotspotLayer) return;

  hotspotLayer.clearLayers();

  if (!Array.isArray(hotspots) || hotspots.length === 0) return;

  hotspots.forEach((p) => {
    if (p.lat == null || p.lon == null) return;

    L.circleMarker([p.lat, p.lon], {
      radius: 5,
      color: "red",
      weight: 1,
      fillColor: "red",
      fillOpacity: 0.8
    })
      .bindPopup(`Hotspot<br/>Stress: ${p.score != null ? p.score.toFixed(3) : "--"}`)
      .addTo(hotspotLayer);
  });
}

/* ===============================
   TREND
================================= */
function drawTrendChart(series) {
  const canvas = document.getElementById("trendChart");
  if (!canvas || !Array.isArray(series) || series.length === 0) return;

  const labels = series.map((x) => x.date);
  const values = series.map((x) => x.stress);

  if (trendChartInstance) {
    trendChartInstance.destroy();
  }

  trendChartInstance = new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Crop Stress",
          data: values,
          borderColor: "#ff4d4d",
          tension: 0.3,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false }
      },
      scales: {
        y: { min: 0, max: 1 }
      }
    }
  });
}

/* ===============================
   TIMELINE
================================= */
async function loadTimelineData() {
  const res = await fetch("/api/analyze-timeseries", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      bbox: lastBBox,
      start_date: document.getElementById("startDate")?.value,
      end_date: document.getElementById("endDate")?.value,
      max_cloud: parseFloat(document.getElementById("maxCloud")?.value || "20")
    })
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  const data = await res.json();

  renderTimeline(data.images || []);
  drawTrendChart(data.stress_series || []);
}

function renderTimeline(data) {
  const container = document.getElementById("timelineContainer");
  if (!container) return;

  container.innerHTML = "";

  if (!Array.isArray(data) || data.length === 0) {
    container.innerHTML = "<div class='timeline-empty'>No timeline data available.</div>";
    return;
  }

  data.forEach((d) => {
    const card = document.createElement("div");
    card.className = "timelineCard";

    const stressText = d.stress != null ? d.stress.toFixed(2) : "--";
    const quicklook = d.quicklook || "";

    card.innerHTML = `
      ${
        quicklook
          ? `<img src="${quicklook}" alt="Timeline quicklook" />`
          : `<div class="timeline-no-image">No image</div>`
      }
      <div class="timelineDate">${d.date || "--"}</div>
      <div class="timelineStress">Stress: ${stressText}</div>
    `;

    if (quicklook) {
      const img = card.querySelector("img");
      if (img) {
        img.addEventListener("click", () => {
          const mainImg = document.getElementById("qlImage");
          if (mainImg) mainImg.src = quicklook + "?t=" + Date.now();
        });
      }
    }

    container.appendChild(card);
  });
}

/* ===============================
   INIT
================================= */
window.addEventListener("DOMContentLoaded", async () => {
  initMap();
  await loadDefaultDates();

  document.getElementById("analyzeBtn")?.addEventListener("click", analyze);
  document.getElementById("clearAoiBtn")?.addEventListener("click", clearAOI);
  document.getElementById("loadModelsBtn")?.addEventListener("click", loadModels);

  bindPixelInspector();
});