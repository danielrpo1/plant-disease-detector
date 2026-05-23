const apiHint = document.getElementById("apiHint");
if (!window.API_URL || window.API_URL.includes("PASTE_")) {
  apiHint.textContent =
    "Vista previa: conecta la API en webapp/config.js después del entrenamiento.";
  apiHint.classList.remove("hidden");
}

const fileInput = document.getElementById("fileInput");
const btnPredict = document.getElementById("btnPredict");
const preview = document.getElementById("preview");
const results = document.getElementById("results");
const loading = document.getElementById("loading");
const errorEl = document.getElementById("error");

let selectedFile = null;

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

  if (!window.API_URL || window.API_URL.includes("PASTE_")) {
    showError("Configura window.API_URL en webapp/config.js");
    return;
  }

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
});

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result;
      const b64 = dataUrl.split(",")[1];
      resolve(b64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function renderResults(data) {
  results.classList.remove("hidden");
  const top = data.predictions[0];
  const pct = (top.confidence * 100).toFixed(1);

  let html = `<h2>${top.display_name}</h2><p class="confidence">Confianza: ${pct}%</p>`;
  html += "<h3>Top 3</h3><ul>";
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
