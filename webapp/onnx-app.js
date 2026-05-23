/** Inferencia ONNX en el navegador (GitHub Pages sin AWS). */
const PlantOnnx = (() => {
  const MEAN = [0.485, 0.456, 0.406];
  const STD = [0.229, 0.224, 0.225];
  let session = null;
  let idxToClass = {};
  let displayEs = {};
  let treatmentTips = {};
  let disclaimer = "";

  function displayName(cls) {
    return displayEs[cls] || cls.replace(/___/g, " - ").replace(/_/g, " ");
  }

  function treatmentFor(cls) {
    return (
      treatmentTips[cls] ||
      "Retire tejido muy dañado, mejore ventilación y consulte extensión agrícola local."
    );
  }

  async function load(modelUrl, metaUrl) {
    if (typeof ort !== "undefined") {
      ort.env.wasm.wasmPaths =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/";
    }
    const [meta, labels, treatments] = await Promise.all([
      fetch(metaUrl).then((r) => r.json()),
      fetch("models/labels_display.json").then((r) => r.json()).catch(() => ({})),
      fetch("models/treatments.json").then((r) => r.json()).catch(() => ({ tips: {} })),
    ]);
    displayEs = labels;
    treatmentTips = treatments.tips || {};
    disclaimer = treatments.disclaimer || "";
    idxToClass = meta.idx_to_class;
    session = await ort.InferenceSession.create(modelUrl, { executionProviders: ["wasm"] });
  }

  function preprocess(img) {
    const s = 224;
    const c = document.createElement("canvas");
    c.width = s;
    c.height = s;
    c.getContext("2d").drawImage(img, 0, 0, s, s);
    const { data } = c.getContext("2d").getImageData(0, 0, s, s);
    const out = new Float32Array(3 * s * s);
    for (let i = 0; i < s * s; i++) {
      const r = data[i * 4] / 255;
      const g = data[i * 4 + 1] / 255;
      const b = data[i * 4 + 2] / 255;
      out[i] = (r - MEAN[0]) / STD[0];
      out[i + s * s] = (g - MEAN[1]) / STD[1];
      out[i + 2 * s * s] = (b - MEAN[2]) / STD[2];
    }
    return new ort.Tensor("float32", out, [1, 3, s, s]);
  }

  async function predict(file) {
    const url = URL.createObjectURL(file);
    const img = await new Promise((res, rej) => {
      const i = new Image();
      i.onload = () => res(i);
      i.onerror = rej;
      i.src = url;
    });
    URL.revokeObjectURL(url);
    const input = preprocess(img);
    const name = session.inputNames[0];
    const logits = (await session.run({ [name]: input }))[session.outputNames[0]].data;
    const max = Math.max(...logits);
    const probs = Array.from(logits).map((x) => Math.exp(x - max));
    const sum = probs.reduce((a, b) => a + b, 0);
    const norm = probs.map((p) => p / sum);
    const top = norm
      .map((p, i) => ({ i, p }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 3)
      .map(({ i, p }) => {
        const cls = idxToClass[String(i)] ?? idxToClass[i];
        return {
          class: cls,
          confidence: Math.round(p * 1e4) / 1e4,
          display_name: displayName(cls),
          treatment_recommendation: treatmentFor(cls),
        };
      });
    return {
      predictions: top,
      top_prediction: top[0].class,
      confidence: top[0].confidence,
      treatment_recommendation: top[0].treatment_recommendation,
      disclaimer,
    };
  }

  return { load, predict };
})();
