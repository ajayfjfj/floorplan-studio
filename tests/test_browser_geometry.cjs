const assert = require('node:assert/strict');
const cv = require('../web/dist/vendor/opencv.js');
const { vectorize } = require('../web/dist/geometry.js');

function run() {
  const width = 64, height = 64;
  const mask = new Uint8Array(width * height);
  const confidence = new Float32Array(width * height).fill(0.8);
  const rectangle = (x1, y1, x2, y2, cls) => {
    for (let y = y1; y <= y2; y++) for (let x = x1; x <= x2; x++) mask[y * width + x] = cls;
  };
  rectangle(5, 5, 26, 26, 2); rectangle(9, 9, 22, 22, 1);
  rectangle(34, 8, 50, 11, 3); rectangle(35, 35, 38, 52, 4);
  const output = vectorize(cv, mask, confidence, width, height, { id: 'synthetic', width: 640, height: 640 });
  assert.deepEqual(output.image, { id: 'synthetic', width: 640, height: 640 });
  assert.equal(output.elements.length, 4);
  assert.equal(output.elements.find(e => e.class === 'wall').holes.length, 1);
  assert(Math.abs(output.elements.find(e => e.class === 'room').area_fraction - 196 / 4096) < 1e-8);
  for (const element of output.elements) {
    assert.equal(element.confidence, 0.8);
    assert(element.polygon.length >= 3);
    for (const point of [...element.polygon, ...element.holes.flat()])
      assert(point.every(value => value >= 0 && value <= 1));
    if (['door', 'window'].includes(element.class)) {
      assert.equal(element.opening_span.length, 2);
      assert(Math.hypot(element.opening_span[0][0] - element.opening_span[1][0],
        element.opening_span[0][1] - element.opening_span[1][1]) > 0.2);
    }
  }
  const empty = vectorize(cv, new Uint8Array(4096), confidence, width, height, { id: 'empty', width, height });
  assert.equal(empty.elements.length, 0);
  console.log('Browser geometry checks passed: components, wall holes, normalized coordinates, opening spans, empty masks.');
}
function check() {
  try { run(); }
  catch (error) { console.error(error.message); process.exitCode = 1; }
}
if (cv.Mat) check(); else cv.onRuntimeInitialized = check;
