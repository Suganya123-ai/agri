let map;
let drawnItems;
let lastBBox = null;
let lastAnalyzeResponse = null;

// Bangalore farmland-ish start
const BANGALORE = [13.05, 77.60];

function setStatus(msg) {
  document.getElementById("status").textContent = "Status: " + msg;
}

function todayUTCISO() {
  const now = new Date();
  const y = now.getUTCFullYear();
  const m = String(now.getUTCMonth() + 1).padStart(2, "0");
  const d = String(now.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

function minusDaysISO(iso, days) {
  const dt = new Date(iso + "T00:00:00Z");
  dt.setUTCDate(dt.getUTCDate() - days);
  const y = dt.getUTCFullYear();
  const m = String(dt.getUTCMonth() + 1).padStart(2, "0");
  const d = String(dt.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

async function loadDefaultDates() {
  const end = todayUTCISO();
  const start = minusDaysISO(end, 20);
  document.getElementById("startDate").value = start;
  document.getElementById("endDate").value = end;
}

function initMap() {
  map = L.map("map").setView(BANGALORE, 11);

  // Satellite imagery (Esri)
  L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    { maxZoom: 19, attribution: "Esri" }
  ).addTo(map);

  // Place names / roads overlay (Esri Reference)
  L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    { maxZoom: 19, attribution: "Esri (Reference)" }
  ).addTo(map);

  // Drawing layer
  drawnItems = new L.FeatureGroup();
  map.addLayer(drawnItems);

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
    const layer = event.layer;
    drawnItems.addLayer(layer);

    const b = layer.getBounds();
    lastBBox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
    setStatus("AOI selected ✅");
  });

  map.on(L.Draw.Event.EDITED, function () {
    drawnItems.eachLayer(layer => {
      const b = layer.getBounds();
      lastBBox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
    });
    setStatus("AOI updated ✅");
  });

  map.on(L.Draw.Event.DELETED, function () {
    lastBBox = null;
    setStatus("AOI cleared");
  });
}

function clearAOI() {
  drawnItems.clearLayers();
  lastBBox = null;
  setStatus("AOI cleared ✅");
}

function updateResults(resp) {
  lastAnalyzeResponse = resp;

  const curr = resp.current_stress;
  const f = resp.forecast;

  document.getElementById("currStress").textContent =
    curr == null ? "--" : `${curr.toFixed(4)} (${Math.round(curr * 100)}%)`;

  document.getElementById("y7").textContent =
    (!f || f.y_7 == null) ? "--" : `${f.y_7.toFixed(4)} (${Math.round(f.y_7 * 100)}%)`;

  document.getElementById("y14").textContent =
    (!f || f.y_14 == null) ? "--" : `${f.y_14.toFixed(4)} (${Math.round(f.y_14 * 100)}%)`;

  document.getElementById("advisory").textContent =
    "Advisory: " + (resp.advisory || "--");

  const layerSelect = document.getElementById("layerSelect");
  const img = document.getElementById("qlImage");

  function showLayer(layerKey) {
    const url = resp.quicklooks[layerKey] || resp.quicklooks["Stress_Analysis"] || resp.quicklooks["True_Color"];
    if (url) img.src = url + "?t=" + Date.now();
  }

  layerSelect.onchange = () => showLayer(layerSelect.value);

  // ✅ default view = Stress_Analysis
  layerSelect.value = resp.quicklooks["Stress_Analysis"] ? "Stress_Analysis" : "True_Color";
  showLayer(layerSelect.value);
}

async function analyze() {
  if (!lastBBox) {
    alert("Please draw a rectangle (AOI) on the map first.");
    return;
  }

  const startDate = document.getElementById("startDate").value;
  const endDate = document.getElementById("endDate").value;
  const maxCloud = parseFloat(document.getElementById("maxCloud").value || "20");

  setStatus("Processing... (this can take a bit)");
  document.getElementById("analyzeBtn").disabled = true;

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

    if (!res.ok) {
      const t = await res.text();
      throw new Error(t);
    }

    const data = await res.json();
    updateResults(data);
    setStatus("Done ✅");
  } catch (e) {
    console.error(e);
    alert("Analyze failed: " + e.message);
    setStatus("Error ❌");
  } finally {
    document.getElementById("analyzeBtn").disabled = false;
  }
}

function bindPixelInspector() {
  const img = document.getElementById("qlImage");

  img.addEventListener("click", async (ev) => {
    if (!lastAnalyzeResponse) {
      alert("Run analysis first.");
      return;
    }

    const rect = img.getBoundingClientRect();
    const rx = ev.clientX - rect.left;
    const ry = ev.clientY - rect.top;

    const nx = rx / rect.width;
    const ny = ry / rect.height;

    const w = img.naturalWidth;
    const h = img.naturalHeight;
    if (!w || !h) return;

    const x = Math.floor(nx * w);
    const y = Math.floor(ny * h);

    const payload = {
      request_id: lastAnalyzeResponse.request_id,
      x, y,
      files: lastAnalyzeResponse.selected_outputs
    };

    try {
      const res = await fetch("/api/pixel-info", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "pixel-info failed");

      document.getElementById("pixXY").textContent = `${data.x}, ${data.y}`;
      document.getElementById("pixClass").textContent = data.class_name;
      document.getElementById("pixNDVI").textContent = data.ndvi == null ? "--" : data.ndvi.toFixed(3);
      document.getElementById("pixNDMI").textContent = data.ndmi == null ? "--" : data.ndmi.toFixed(3);
      document.getElementById("pixNDRE").textContent = data.ndre == null ? "--" : data.ndre.toFixed(3);
      document.getElementById("pixStress").textContent = data.stress == null ? "--" : `${data.stress.toFixed(3)} (${Math.round(data.stress * 100)}%)`;

      document.getElementById("expStatus").textContent = data.explain.status;
      document.getElementById("expWhy").textContent = data.explain.why;
      document.getElementById("expSol").textContent = data.explain.solution;
    } catch (e) {
      console.error(e);
      alert("Pixel inspector failed: " + e.message);
    }
  });
}

async function loadModels() {
  const ul = document.getElementById("modelsList");
  ul.innerHTML = "";

  try {
    const res = await fetch("/api/models");
    const data = await res.json();

    if (!data.models || data.models.length === 0) {
      ul.innerHTML = "<li>No models found in /models</li>";
      return;
    }

    data.models.forEach(m => {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = m.url;
      a.textContent = m.name;
      a.target = "_blank";
      li.appendChild(a);
      ul.appendChild(li);
    });
  } catch (e) {
    ul.innerHTML = "<li>Error loading models</li>";
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  initMap();
  await loadDefaultDates();

  document.getElementById("analyzeBtn").addEventListener("click", analyze);
  document.getElementById("clearAoiBtn").addEventListener("click", clearAOI);
  document.getElementById("loadModelsBtn").addEventListener("click", loadModels);

  bindPixelInspector();
});