/* Inference stays off the UI thread; image pixels never leave the browser. */
let sessionPromise, cvPromise;
const status = message => self.postMessage({ type: 'status', message });

async function modelSession() {
  if (!sessionPromise) sessionPromise = (async () => {
    status('Loading model · first run downloads about 36 MB');
    importScripts('vendor/ort.wasm.min.js');
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = new URL('vendor/', self.location.href).href;
    return ort.InferenceSession.create(new URL('models/floorplan-segformer.onnx', self.location.href).href,
      { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  })().catch(error => { sessionPromise = null; throw error; });
  return sessionPromise;
}

async function openCV() {
  if (!cvPromise) cvPromise = (async () => {
    importScripts('vendor/opencv.js', 'geometry.js');
    if (self.cv instanceof Promise) self.cv = await self.cv;
    if (!self.cv.Mat) await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Geometry runtime timed out. Please retry.')), 90000);
      self.cv.onRuntimeInitialized = () => { clearTimeout(timer); resolve(); };
      self.cv.onAbort = message => { clearTimeout(timer); reject(new Error(String(message))); };
    });
    // Emscripten's Module has a self-returning .then(); do not resolve it as a Promise.
    return { runtime: self.cv };
  })().catch(error => { cvPromise = null; throw error; });
  return cvPromise;
}

function bilinear(data, offset, size, x, y) {
  x = Math.max(0, Math.min(size - 1, x)); y = Math.max(0, Math.min(size - 1, y));
  const x0 = Math.floor(x), y0 = Math.floor(y), x1 = Math.min(size - 1, x0 + 1), y1 = Math.min(size - 1, y0 + 1);
  const fx = x - x0, fy = y - y0;
  return (1 - fy) * ((1 - fx) * data[offset + y0 * size + x0] + fx * data[offset + y0 * size + x1]) +
    fy * ((1 - fx) * data[offset + y1 * size + x0] + fx * data[offset + y1 * size + x1]);
}

function decode(logits, box, image) {
  const expanded = new Float32Array(5 * 512 * 512);
  for (let c = 0; c < 5; c++) for (let y = 0; y < 512; y++) for (let x = 0; x < 512; x++)
    expanded[c * 512 * 512 + y * 512 + x] = bilinear(logits, c * 128 * 128, 128, (x + 0.5) / 4 - 0.5, (y + 0.5) / 4 - 0.5);
  const scale = Math.min(1, 1024 / Math.max(image.width, image.height));
  const width = Math.max(2, Math.round(image.width * scale)), height = Math.max(2, Math.round(image.height * scale));
  const mask = new Uint8Array(width * height), confidence = new Float32Array(width * height), values = new Float32Array(5);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const sx = box.left + Math.max(0, Math.min(box.width - 1, (x + 0.5) * box.width / width - 0.5));
    const sy = box.top + Math.max(0, Math.min(box.height - 1, (y + 0.5) * box.height / height - 0.5));
    let best = 0, max = -Infinity, sum = 0;
    for (let c = 0; c < 5; c++) {
      values[c] = bilinear(expanded, c * 512 * 512, 512, sx, sy);
      if (values[c] > max) { max = values[c]; best = c; }
    }
    for (let c = 0; c < 5; c++) sum += Math.exp(values[c] - max);
    mask[y * width + x] = best; confidence[y * width + x] = 1 / sum;
  }
  return { mask, confidence, width, height };
}

self.onmessage = async ({ data }) => {
  const { pixels, box, image } = data;
  try {
    const session = await modelSession();
    status('Detecting walls, doors, windows, and rooms…');
    const start = performance.now();
    const input = new ort.Tensor('float32', pixels, [1, 3, 512, 512]);
    let result;
    try { result = await session.run({ pixel_values: input }); }
    finally { input.dispose(); }
    const decoded = decode(result.logits.data, box, image);
    result.logits.dispose();
    status('Tracing polygons and opening spans…');
    const { runtime: cv } = await openCV();
    const prediction = FloorplanGeometry.vectorize(cv, decoded.mask, decoded.confidence, decoded.width, decoded.height, image);
    self.postMessage({ type: 'result', prediction, elapsed: performance.now() - start,
      width: decoded.width, height: decoded.height, mask: decoded.mask }, [decoded.mask.buffer]);
  } catch (error) {
    self.postMessage({ type: 'error', message: error.message || String(error) });
  }
};
