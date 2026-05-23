const apiHint = document.getElementById("apiHint");
const useApi = window.API_URL && !window.API_URL.includes("PASTE_");
const useOnnx = !useApi;

let onnxReady = false;

if (useOnnx) {
  apiHint.textContent = "Cargando modelo en el navegador…";
  apiHint.classList.remove("hidden");
}

const fileInput = document.getElementById("fileInput");
const btnPredict = document.getElementById("btnPredict");
const preview = document.getElementById("preview");
const results = document.getElementById("results");
const loading = document.getElementById("loading");
const errorEl = document.getElementById("error");

let selectedFile = null;

async function initOnnx() {
  if (!useOnnx) return;
  try {
    await PlantOnnx.load(window.ONNX_MODEL_URL, window.ONNX_META_URL);
    onnxReady = true;
    apiHint.textContent = "Modo demo: inferencia ONNX en tu navegador (sin servidor).";
  } catch (e) {
    apiHint.textContent = "Modelo aún no disponible. Entrenamiento en curso…";
    console.error(e);
  }
}

initOnnx();

fileInput.addEventListener("change", () => {
  selectedFile = fileInput.files[0] || null;
  errorEl.classList.add("hidden");
  results.classList.add("hidden");

  if (!selectedFile) {
    btnPredict.disabled = true;
    preview.classList.add("hidden");
    return;
  }

  btnPredict.disabled = false;
  preview.classList.remove("hidden");
  preview.innerHTML = "";
  const img = document.createElement("img");
  img.src = URL.createObjectURL(selectedFile);
  img.alt = "Vista previa";
  preview.appendChild(img);
});

btnPredict.addEventListener("click", async () => {
  if (!selectedFile) return;

  if (useApi) {
    await predictViaApi();
  } else {
    if (!onnxReady) {
      showError("El modelo ONNX aún no está en GitHub Pages. Vuelve en unos minutos.");
      return;
    }
    await predictViaOnnx();
  }
});

async function predictViaApi() {
  loading.classList.remove("hidden");
  results.classList.add("hidden");
  errorEl.classList.add("hidden");
  btnPredict.disabled = true;
  try {
    const base64 = await fileToBase64(selectedFile);
    const res = await fetch(window.API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: base64 }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    renderResults(data);
  } catch (err) {
    showError(err.message);
  } finally {
    loading.classList.add("hidden");
    btnPredict.disabled = false;
  }
}

async function predictViaOnnx() {
  loading.classList.remove("hidden");
  loading.textContent = "Analizando (ONNX local)…";
  results.classList.add("hidden");
  errorEl.classList.add("hidden");
  btnPredict.disabled = true;
  try {
    const data = await PlantOnnx.predict(selectedFile);
    renderResults(data);
  } catch (err) {
    showError(err.message);
  } finally {
    loading.classList.add("hidden");
    loading.textContent = "Analizando…";
    btnPredict.disabled = false;
  }
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function renderResults(data) {
  results.classList.remove("hidden");
  const top = data.predictions[0];
  const pct = (top.confidence * 100).toFixed(1);
  let html = `<h2>${top.display_name}</h2><p class="confidence">Confianza: ${pct}%</p><h3>Top 3</h3><ul>`;
  for (const p of data.predictions) {
    html += `<li>${p.display_name} — ${(p.confidence * 100).toFixed(1)}%</li>`;
  }
  html += "</ul>";
  results.innerHTML = html;
}

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.classList.remove("hidden");
}
