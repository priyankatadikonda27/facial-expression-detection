// On-device inference: YuNet face detection + eye alignment + ResNet18 (RAF-DB), mirroring the Python pipeline.
const ENGINE = (() => {
  const LABELS = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"]; // model output order
  const MARGIN = -0.05, SHIFT = -0.05;            // crop calibrated by calibrate.py (models/pipeline.json)
  const DET_THR = 0.7, NMS = 0.3, MAX_FACES = 4, S = 640, IMG = 224;
  const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
  let det, fer;

  const frame = document.createElement("canvas"), fctx = frame.getContext("2d", { willReadFrequently: true });
  const crop = document.createElement("canvas"); crop.width = crop.height = IMG;
  const cctx = crop.getContext("2d", { willReadFrequently: true });
  const heat = document.createElement("canvas"); heat.width = heat.height = 7;
  const hctx = heat.getContext("2d");
  const out = document.createElement("canvas"); out.width = out.height = IMG;
  const octx = out.getContext("2d");

  async function fetchModel(url, onProgress) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Could not download ${url} (${r.status})`);
    const total = +r.headers.get("content-length") || 0;
    if (!r.body || !total) return new Uint8Array(await r.arrayBuffer());
    const reader = r.body.getReader(), parts = []; let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value); got += value.length;
      onProgress && onProgress(Math.round(100 * got / total));
    }
    const buf = new Uint8Array(got); let o = 0;
    for (const p of parts) { buf.set(p, o); o += p.length; }
    return buf;
  }

  async function load(onStatus) {
    ort.env.wasm.wasmPaths = new URL("ort/", location.href).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
    onStatus("Loading face detector…");
    det = await ort.InferenceSession.create(await fetchModel("models/face_detection_yunet_2023mar.onnx"), opts);
    const model = await fetchModel("models/fer_resnet18.onnx", p => onStatus(`Loading AI model… ${p}% (45 MB, first visit only)`));
    onStatus("Starting AI model…");
    fer = await ort.InferenceSession.create(model, opts);
  }

  const iou = (a, b) => {
    const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
    const x2 = Math.min(a[0] + a[2], b[0] + b[2]), y2 = Math.min(a[1] + a[3], b[1] + b[3]);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    return inter / (a[2] * a[3] + b[2] * b[3] - inter);
  };

  async function detect() {
    const w = frame.width, h = frame.height, px = fctx.getImageData(0, 0, w, h).data, P = S * S;
    const t = new Float32Array(3 * P);                      // BGR, 0-255, zero-padded to 640x640
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4, j = y * S + x;
      t[j] = px[i + 2]; t[P + j] = px[i + 1]; t[2 * P + j] = px[i];
    }
    const o = await det.run({ input: new ort.Tensor("float32", t, [1, 3, S, S]) });
    const dets = [];
    for (const st of [8, 16, 32]) {
      const cls = o["cls_" + st].data, obj = o["obj_" + st].data, bb = o["bbox_" + st].data, kp = o["kps_" + st].data, cols = S / st;
      for (let i = 0; i < cls.length; i++) {
        const sc = Math.sqrt(Math.min(Math.max(cls[i], 0), 1) * Math.min(Math.max(obj[i], 0), 1));
        if (sc < DET_THR) continue;
        const r = Math.floor(i / cols), c = i % cols;
        const cx = (c + bb[4 * i]) * st, cy = (r + bb[4 * i + 1]) * st;
        const bw = Math.exp(bb[4 * i + 2]) * st, bh = Math.exp(bb[4 * i + 3]) * st;
        dets.push({ score: sc, box: [cx - bw / 2, cy - bh / 2, bw, bh],
          eyes: [[(kp[10 * i] + c) * st, (kp[10 * i + 1] + r) * st], [(kp[10 * i + 2] + c) * st, (kp[10 * i + 3] + r) * st]] });
      }
    }
    dets.sort((a, b) => b.score - a.score);
    const keep = [];
    for (const d of dets) if (keep.every(q => iou(d.box, q.box) <= NMS)) keep.push(d);
    return keep;
  }

  // Same affine as cv2.getRotationMatrix2D + recentre in common.align_face
  function align(f) {
    const [x, y, w, h] = f.box, cx = x + w / 2, cy = y + h / 2 + SHIFT * h;
    const [[rx, ry], [lx, ly]] = f.eyes, ang = Math.atan2(ly - ry, lx - rx);
    const sc = IMG / (Math.max(w, h) * (1 + 2 * MARGIN)), a = sc * Math.cos(ang), b = sc * Math.sin(ang);
    const e = (1 - a) * cx - b * cy + IMG / 2 - cx, g = b * cx + (1 - a) * cy + IMG / 2 - cy;
    cctx.setTransform(1, 0, 0, 1, 0, 0); cctx.fillStyle = "rgb(120,120,120)"; cctx.fillRect(0, 0, IMG, IMG);
    cctx.setTransform(a, -b, b, a, e, g); cctx.drawImage(frame, 0, 0); cctx.setTransform(1, 0, 0, 1, 0, 0);
    return cctx.getImageData(0, 0, IMG, IMG);
  }

  async function classify(crops) {                            // each face + its mirror image (test-time augmentation)
    const P = IMG * IMG, t = new Float32Array(crops.length * 2 * 3 * P);
    crops.forEach((im, k) => {
      const px = im.data, o = 2 * k * 3 * P, of = (2 * k + 1) * 3 * P;
      for (let y = 0; y < IMG; y++) for (let x = 0; x < IMG; x++) {
        const i = (y * IMG + x) * 4, j = y * IMG + x, jf = y * IMG + (IMG - 1 - x);
        for (let ch = 0; ch < 3; ch++) { const v = (px[i + ch] / 255 - MEAN[ch]) / STD[ch]; t[o + ch * P + j] = v; t[of + ch * P + jf] = v; }
      }
    });
    const o = await fer.run({ input: new ort.Tensor("float32", t, [crops.length * 2, 3, IMG, IMG]) });
    return { p: o.probs.data, cams: o.cams.data };
  }

  function jet(v) {
    const c = q => Math.round(255 * Math.min(Math.max(1.5 - Math.abs(4 * v - q), 0), 1));
    return [c(3), c(2), c(1)];
  }

  function camImage(cropData, cams) {                         // Grad-CAM of the unflipped main face
    const p = Array.from({ length: 7 }, (_, k) => k), top = p.reduce((a, b) => (cams.top[b] > cams.top[a] ? b : a));
    const m = cams.data.slice(top * 49, top * 49 + 49).map(v => Math.max(v, 0)), mx = Math.max(...m) + 1e-8;
    const im = hctx.createImageData(7, 7);
    m.forEach((v, i) => { const [r, g, b] = jet(v / mx); im.data.set([r, g, b, 255], i * 4); });
    hctx.putImageData(im, 0, 0);
    octx.putImageData(cropData, 0, 0);
    octx.globalAlpha = 0.45; octx.imageSmoothingEnabled = true; octx.drawImage(heat, 0, 0, IMG, IMG); octx.globalAlpha = 1;
    return { label: LABELS[top], url: out.toDataURL("image/png") };
  }

  // Analyse whatever is drawn on `source` (video or image). Returns the same JSON shape as the old /predict API.
  async function analyse(source, sw, sh, wantCam) {
    const sc = Math.min(480 / sw, 640 / sh);
    frame.width = Math.round(sw * sc); frame.height = Math.round(sh * sc);
    fctx.drawImage(source, 0, 0, frame.width, frame.height);
    const faces = (await detect()).sort((a, b) => b.box[2] * b.box[3] - a.box[2] * a.box[3]).slice(0, MAX_FACES);
    const res = { size: [frame.width, frame.height], faces: [] };
    if (!faces.length) return res;
    const crops = faces.map(align), { p, cams } = await classify(crops);
    faces.forEach((f, k) => {
      const probs = {};
      LABELS.forEach((n, c) => { probs[n] = (p[2 * k * 7 + c] + p[(2 * k + 1) * 7 + c]) / 2; });
      res.faces.push({ box: f.box.map(Math.round), probs });
    });
    if (wantCam) {
      const c = camImage(crops[0], { data: cams.slice(0, 343), top: Array.from(p.slice(0, 7)) });
      res.faces[0].cam = c.url; res.faces[0].cam_label = c.label;
    }
    return res;
  }

  return { load, analyse, LABELS };
})();
