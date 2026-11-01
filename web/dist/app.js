const $ = id => document.getElementById(id);
const classes = [
  { id: 'room', label: 'Room regions', color: '#91c9d3', iou: 91.93 },
  { id: 'wall', label: 'Wall regions', color: '#475569', iou: 70.05 },
  { id: 'door', label: 'Door regions', color: '#ee8b48', iou: 49.66 },
  { id: 'window', label: 'Window regions', color: '#56a971', iou: 67.50 },
];
let currentImage, currentPrediction, currentName = 'floorplan', currentSample;
let worker, busy = false, selection = 0, temporaryURL, inferenceTimer;
const visible = new Set(classes.map(c => c.id));
const svgNS = 'http://www.w3.org/2000/svg';

for (const cls of classes) {
  const label = document.createElement('label');
  label.className = 'class-row'; label.style.setProperty('--class-color', cls.color);
  label.title = 'Filter the preview; JSON download always includes all detected regions.';
  const input = document.createElement('input'); input.type = 'checkbox'; input.checked = true;
  input.setAttribute('aria-label', `Show ${cls.label.toLowerCase()}`);
  input.addEventListener('change', () => {
    input.checked ? visible.add(cls.id) : visible.delete(cls.id); renderOverlay();
  });
  const text = document.createElement('span'); text.textContent = cls.label;
  const count = document.createElement('strong'); count.id = `count-${cls.id}`; count.textContent = '—';
  label.append(input, text, count); $('class-list').append(label);
  const row = document.createElement('div'); row.className = 'metric-row';
  const name = document.createElement('span'); name.textContent = cls.id[0].toUpperCase() + cls.id.slice(1);
  const track = document.createElement('div'); track.className = 'metric-track';
  const bar = document.createElement('span'); bar.style.cssText = `width:${cls.iou}%;background:${cls.color}`;
  track.append(bar);
  const value = document.createElement('strong'); value.textContent = `${cls.iou.toFixed(2)}%`;
  row.append(name, track, value); $('metric-bars').append(row);
}

function status(message, kind = '') {
  $('status').textContent = message; $('status-dot').className = `status-dot ${kind}`;
}
function error(message) {
  $('error-box').textContent = message; $('error-box').hidden = false;
  status('Unable to complete detection. You can retry or choose another image.', 'error');
}
function clearError() { $('error-box').hidden = true; }
function setBusy(value) {
  if (!value) clearTimeout(inferenceTimer);
  busy = value; $('run-button').disabled = value || !currentImage;
  $('run-button').innerHTML = value ? 'Running detection…' : '<span aria-hidden="true">↗</span> Run detection';
  $('file-input').disabled = value;
  $('download-sample').disabled = value || !currentSample || !currentImage;
  $('drop-zone').setAttribute('aria-disabled', String(value));
  for (const button of document.querySelectorAll('.sample-card')) button.disabled = value;
}
function enableExports() {
  const disabled = !currentPrediction;
  for (const id of ['download-json', 'download-overlay', 'download-overview', 'copy-json']) $(id).disabled = disabled;
}

function ringPath(points) {
  return points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0] * 1000},${p[1] * 1000}`).join(' ') + 'Z';
}
function renderOverlay() {
  const svg = $('geometry-overlay'); svg.replaceChildren();
  svg.hidden = !$('overlay-toggle').checked;
  const opacity = Number($('opacity').value) / 100;
  for (const element of currentPrediction?.elements || []) {
    if (!visible.has(element.class)) continue;
    const color = classes.find(c => c.id === element.class).color;
    const path = document.createElementNS(svgNS, 'path');
    path.setAttribute('d', [ringPath(element.polygon), ...element.holes.map(ringPath)].join(' '));
    path.setAttribute('fill', color); path.setAttribute('fill-opacity', opacity);
    path.setAttribute('fill-rule', 'evenodd'); path.setAttribute('stroke', color);
    path.setAttribute('stroke-width', '.65'); path.setAttribute('stroke-opacity', '.65');
    path.setAttribute('tabindex', '0'); path.setAttribute('role', 'button');
    path.setAttribute('aria-label', `Inspect ${element.id}`);
    const title = document.createElementNS(svgNS, 'title'); title.textContent = element.id; path.append(title);
    const select = () => {
      svg.querySelector('.selected')?.classList.remove('selected'); path.classList.add('selected');
      $('element-details').hidden = false; $('element-name').textContent = element.id;
      $('element-info').textContent = `${element.polygon.length} vertices · ${(element.area_fraction * 100).toFixed(2)}% of image` +
        (element.confidence === null ? '' : ` · ${(element.confidence * 100).toFixed(1)}% mean probability (uncalibrated)`);
    };
    path.addEventListener('click', select);
    path.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); } });
    svg.append(path);
  }
}
function showPrediction(prediction, source) {
  currentPrediction = prediction;
  $('total-count').textContent = prediction ? prediction.elements.length : '—';
  $('result-source').textContent = source;
  $('element-details').hidden = true;
  for (const cls of classes) $('count-' + cls.id).textContent = prediction ? prediction.elements.filter(e => e.class === cls.id).length : '—';
  $('json-code').textContent = prediction ? JSON.stringify(prediction, null, 2) : 'Run detection to generate spatial JSON for this image.';
  enableExports(); renderOverlay();
}
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image(); image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('This image could not be opened. Try a PNG, JPG, or WebP.'));
    image.src = src;
  });
}
function displayImage(image, name) {
  currentImage = image; currentName = name;
  $('plan-image').src = image.src; $('plan-image').alt = `Floor plan: ${name}`;
  $('drawing').style.aspectRatio = `${image.naturalWidth} / ${image.naturalHeight}`;
  // Keep unusually tall plans inside the workbench without stretching them.
  $('drawing').style.maxWidth = `${Math.min(560, 560 * image.naturalWidth / image.naturalHeight)}px`;
  $('dimensions').textContent = `${image.naturalWidth} × ${image.naturalHeight} px`;
  $('run-button').disabled = false;
  $('download-sample').disabled = !currentSample;
}
function selectSampleButton(id) {
  for (const button of document.querySelectorAll('.sample-card')) button.setAttribute('aria-pressed', String(button.dataset.id === id));
}
async function useSample(sample) {
  if (busy) return;
  const token = ++selection; clearError(); status('Opening sample plan…', 'busy');
  $('run-button').disabled = true;
  $('download-sample').disabled = true;
  showPrediction(null, 'Ready for detection');
  try {
    const image = await loadImage(`samples/${sample.id}/floorplan.png`);
    if (token !== selection) return;
    if (temporaryURL) { URL.revokeObjectURL(temporaryURL); temporaryURL = undefined; }
    currentSample = sample; selectSampleButton(sample.id); displayImage(image, sample.title);
    showPrediction(null, 'Ready for detection'); showTab('visual');
    status(`${sample.title} · ready. Select Run detection to generate a new result.`);
  } catch (err) { if (token === selection) { $('run-button').disabled = !currentImage; error(err.message); } }
}
async function useFile(file) {
  if (!file || busy) return;
  clearError();
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) { error('Choose a PNG, JPG, or WebP image. PDF and SVG are not supported.'); return; }
  if (file.size > 10 * 1024 * 1024) { error('This file is larger than 10 MB. Please resize it and try again.'); return; }
  const token = ++selection;
  $('run-button').disabled = true;
  $('download-sample').disabled = true;
  showPrediction(null, 'Ready for detection');
  const url = URL.createObjectURL(file);
  try {
    const image = await loadImage(url);
    if (token !== selection) { URL.revokeObjectURL(url); return; }
    if (image.naturalWidth < 2 || image.naturalHeight < 2 || image.naturalWidth * image.naturalHeight > 16000000)
      throw new Error('Use an image between 2 × 2 pixels and 16 megapixels. Resize large scans first.');
    if (temporaryURL) URL.revokeObjectURL(temporaryURL);
    temporaryURL = url; currentSample = undefined; selectSampleButton(''); displayImage(image, file.name);
    showPrediction(null, 'Ready for detection'); showTab('visual');
    status(`${file.name} · ready. Select Run detection to extract the geometry.`);
  } catch (err) { URL.revokeObjectURL(url); if (token === selection) { $('run-button').disabled = !currentImage; error(err.message); } }
}

function prepareImage(image) {
  const canvas = document.createElement('canvas'); canvas.width = 512; canvas.height = 512;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.fillStyle = '#fff'; context.fillRect(0, 0, 512, 512);
  const scale = Math.min(512 / image.naturalWidth, 512 / image.naturalHeight);
  const width = Math.max(1, Math.round(image.naturalWidth * scale)), height = Math.max(1, Math.round(image.naturalHeight * scale));
  const left = Math.floor((512 - width) / 2), top = Math.floor((512 - height) / 2);
  context.drawImage(image, left, top, width, height);
  const rgba = context.getImageData(0, 0, 512, 512).data;
  const pixels = new Float32Array(3 * 512 * 512), mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  for (let p = 0; p < 512 * 512; p++) for (let c = 0; c < 3; c++) pixels[c * 512 * 512 + p] = (rgba[p * 4 + c] / 255 - mean[c]) / std[c];
  return { pixels, box: { left, top, width, height } };
}
function runDetection() {
  if (busy || !currentImage) return;
  clearError(); setBusy(true); showTab('visual');
  showPrediction(null, 'Running detection…');
  try {
    const prepared = prepareImage(currentImage);
    if (!worker) {
      worker = new Worker(new URL('inference-worker.js', import.meta.url));
      worker.onmessage = ({ data }) => {
        if (data.type === 'status') status(data.message, 'busy');
        if (data.type === 'result') {
          showPrediction(data.prediction, `Live browser prediction · ${(data.elapsed / 1000).toFixed(1)}s`);
          status(`Detection complete · ${data.prediction.elements.length} regions · ${data.width} × ${data.height} geometry raster. Click a region to inspect it.`);
          setBusy(false);
        }
        if (data.type === 'error') { error(data.message); setBusy(false); }
      };
      worker.onerror = () => { error('The browser inference engine stopped. Refresh and retry in a recent Chrome, Edge, Firefox, or Safari browser.'); worker.terminate(); worker = undefined; setBusy(false); };
    }
    status('Preparing browser inference…', 'busy');
    inferenceTimer = setTimeout(() => {
      worker?.terminate(); worker = undefined; setBusy(false);
      error('Detection took too long. Check your connection, resize large images, and retry. A recent desktop browser usually runs faster.');
    }, 150000);
    worker.postMessage({ ...prepared, image: { id: currentSample?.source_id || currentName,
      width: currentImage.naturalWidth, height: currentImage.naturalHeight } }, [prepared.pixels.buffer]);
  } catch (err) { error(err.message); setBusy(false); }
}

function showTab(name) {
  for (const tab of ['visual', 'json']) {
    $('tab-' + tab).setAttribute('aria-selected', String(tab === name));
    $('tab-' + tab).tabIndex = tab === name ? 0 : -1;
    $('view-' + tab).hidden = tab !== name;
  }
}
for (const tab of ['visual', 'json']) {
  $('tab-' + tab).addEventListener('click', () => showTab(tab));
  $('tab-' + tab).addEventListener('keydown', event => {
    if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      event.preventDefault(); const next = event.key === 'Home' ? 'visual' : event.key === 'End' ? 'json' : tab === 'visual' ? 'json' : 'visual';
      showTab(next); $('tab-' + next).focus();
    }
  });
}
const filename = () => currentName.replace(/\.[^.]+$/, '').replace(/[^a-z0-9_-]+/gi, '_') || 'floorplan';
function download(blob, name) {
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 10000);
}
$('download-json').addEventListener('click', () => {
  if (currentPrediction) download(new Blob([JSON.stringify(currentPrediction, null, 2) + '\n'], { type: 'application/json' }), `${filename()}_prediction.json`);
});
$('download-sample').addEventListener('click', async () => {
  if (!currentSample || busy) return;
  const sample = currentSample;
  try {
    const response = await fetch(new URL(`samples/${sample.id}/floorplan.png`, import.meta.url));
    if (!response.ok) throw new Error('The sample image could not be downloaded. Please retry.');
    download(await response.blob(), `${sample.id}_floorplan.png`);
  } catch (err) { error(err.message); }
});
$('copy-json').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(JSON.stringify(currentPrediction, null, 2)); $('copy-json').textContent = 'Copied'; setTimeout(() => { $('copy-json').textContent = 'Copy JSON'; }, 2000); }
  catch { status('Clipboard permission unavailable. Use Download JSON to save the complete output.'); }
});
function downloadOverview() {
  if (!currentPrediction || !currentImage) return;
  const scale = Math.min(1, 2048 / Math.max(currentImage.naturalWidth, currentImage.naturalHeight));
  const imageWidth = Math.max(1, Math.round(currentImage.naturalWidth * scale));
  const imageHeight = Math.max(1, Math.round(currentImage.naturalHeight * scale));
  const canvas = document.createElement('canvas'); canvas.width = Math.max(720, imageWidth + 48); canvas.height = imageHeight + 176;
  const ctx = canvas.getContext('2d'), offsetX = (canvas.width - imageWidth) / 2, offsetY = 96;
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#243b36'; ctx.font = '600 23px system-ui, sans-serif'; ctx.fillText('Floorplan visual overview', 24, 36);
  ctx.fillStyle = '#66766f'; ctx.font = '13px system-ui, sans-serif';
  const label = currentName.length > 75 ? currentName.slice(0, 72) + '…' : currentName;
  ctx.fillText(label, 24, 61, canvas.width - 48);
  ctx.font = '11px system-ui, sans-serif'; ctx.fillText('Fine-tuned SegFormer-B0 · Live detection · ' + currentPrediction.elements.length + ' regions', 24, 80);
  ctx.drawImage(currentImage, offsetX, offsetY, imageWidth, imageHeight);
  if ($('overlay-toggle').checked) for (const e of currentPrediction.elements) {
    if (!visible.has(e.class)) continue;
    ctx.beginPath();
    for (const ring of [e.polygon, ...e.holes]) {
      ring.forEach((p, i) => i ? ctx.lineTo(offsetX + p[0] * (imageWidth - 1), offsetY + p[1] * (imageHeight - 1)) : ctx.moveTo(offsetX + p[0] * (imageWidth - 1), offsetY + p[1] * (imageHeight - 1)));
      ctx.closePath();
    }
    ctx.fillStyle = classes.find(c => c.id === e.class).color; ctx.globalAlpha = Number($('opacity').value) / 100; ctx.fill('evenodd');
    ctx.globalAlpha = 0.7; ctx.lineWidth = 0.65; ctx.strokeStyle = ctx.fillStyle; ctx.stroke();
  }
  ctx.globalAlpha = 1;
  classes.forEach((cls, i) => {
    const x = 24 + i * (canvas.width - 48) / 4, y = offsetY + imageHeight + 30;
    const shown = $('overlay-toggle').checked && visible.has(cls.id);
    ctx.fillStyle = shown ? cls.color : '#dce2dd'; ctx.fillRect(x, y - 10, 11, 11);
    ctx.fillStyle = '#243b36'; ctx.font = '11px system-ui, sans-serif';
    const value = shown ? currentPrediction.elements.filter(e => e.class === cls.id).length : 'hidden';
    ctx.fillText(`${cls.label}: ${value}`, x + 17, y, (canvas.width - 48) / 4 - 20);
  });
  ctx.fillStyle = '#66766f'; ctx.font = '10px system-ui, sans-serif';
  ctx.fillText('Review merged regions before 3D use. Real-world scale and heights require separate calibration.', 24, canvas.height - 18, canvas.width - 48);
  canvas.toBlob(blob => { if (blob) download(blob, `${filename()}_overview.png`); }, 'image/png');
}
$('download-overlay').addEventListener('click', downloadOverview);
$('download-overview').addEventListener('click', downloadOverview);
$('overlay-toggle').addEventListener('change', renderOverlay);
$('opacity').addEventListener('input', () => { $('opacity-value').textContent = `${$('opacity').value}%`; renderOverlay(); });
$('run-button').addEventListener('click', runDetection);
$('file-input').addEventListener('change', event => { useFile(event.target.files[0]); event.target.value = ''; });
$('drop-zone').addEventListener('keydown', event => { if ((event.key === 'Enter' || event.key === ' ') && !busy) { event.preventDefault(); $('file-input').click(); } });
for (const type of ['dragenter', 'dragover']) $('drop-zone').addEventListener(type, event => { event.preventDefault(); if (!busy) $('drop-zone').classList.add('dragging'); });
for (const type of ['dragleave', 'drop']) $('drop-zone').addEventListener(type, event => { event.preventDefault(); $('drop-zone').classList.remove('dragging'); if (type === 'drop') useFile(event.dataTransfer.files[0]); });
window.addEventListener('dragover', event => event.preventDefault());
window.addEventListener('drop', event => event.preventDefault());

try {
  const samples = [];
  for (const button of document.querySelectorAll('#sample-grid .sample-card')) {
    const sample = { id: button.dataset.id, title: button.dataset.title, source_id: button.dataset.sourceId };
    samples.push(sample);
    button.addEventListener('click', () => useSample(sample));
  }
  if (!samples.length) throw new Error('Sample images are unavailable. You can upload your own image.');
  await useSample(samples[0]);
} catch (err) { error(err.message); }
