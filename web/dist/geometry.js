/* Connected-component geometry shared by the inference worker and Node checks. */
(function (root) {
  const CLASSES = ['background', 'room', 'wall', 'door', 'window'];
  const round = (n, digits = 6) => Number(n.toFixed(digits));

  function vectorize(cv, mask, confidence, width, height, image) {
    const elements = [];
    const minimumArea = Math.max(2, Math.round(12 * width * height / (512 * 512)));
    const point = (x, y) => [round(Math.max(0, Math.min(1, x / (width - 1)))),
      round(Math.max(0, Math.min(1, y / (height - 1))))];
    const ring = contour => {
      const simplified = new cv.Mat();
      try {
        cv.approxPolyDP(contour, simplified, Math.max(0.5, 0.001 * Math.max(width, height)), true);
        const points = [];
        for (let i = 0; i < simplified.data32S.length; i += 2)
          points.push(point(simplified.data32S[i], simplified.data32S[i + 1]));
        return points;
      } finally { simplified.delete(); }
    };
    for (let cls = 1; cls < 5; cls++) {
      const binary = cv.Mat.zeros(height, width, cv.CV_8UC1);
      const labels = new cv.Mat(), stats = new cv.Mat(), centroids = new cv.Mat();
      try {
        for (let p = 0; p < mask.length; p++) binary.data[p] = mask[p] === cls ? 1 : 0;
        const count = cv.connectedComponentsWithStats(binary, labels, stats, centroids, 8, cv.CV_32S);
        let classCount = 0;
        for (let component = 1; component < count; component++) {
          const area = stats.intAt(component, cv.CC_STAT_AREA);
          if (area < minimumArea) continue;
          const region = cv.Mat.zeros(height, width, cv.CV_8UC1);
          const contours = new cv.MatVector(), hierarchy = new cv.Mat();
          let confidenceSum = 0;
          try {
            for (let p = 0; p < mask.length; p++) {
              if (labels.data32S[p] === component) {
                region.data[p] = 1;
                confidenceSum += confidence[p];
              }
            }
            cv.findContours(region, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
            for (let i = 0; i < contours.size(); i++) {
              if (hierarchy.data32S[i * 4 + 3] !== -1) continue;
              const contour = contours.get(i);
              try {
                const polygon = ring(contour);
                if (polygon.length < 3) continue;
                const holes = [];
                for (let j = 0; j < contours.size(); j++) {
                  if (hierarchy.data32S[j * 4 + 3] !== i) continue;
                  const child = contours.get(j);
                  try { const hole = ring(child); if (hole.length >= 3) holes.push(hole); }
                  finally { child.delete(); }
                }
                const b = cv.boundingRect(contour);
                const item = {
                  id: `${CLASSES[cls]}_${String(++classCount).padStart(3, '0')}`,
                  class: CLASSES[cls], polygon, holes,
                  bbox: [...point(b.x, b.y), ...point(b.x + b.width - 1, b.y + b.height - 1)],
                  area_fraction: round(area / (width * height), 8),
                  confidence: round(confidenceSum / area, 5),
                };
                if (cls === 3 || cls === 4) {
                  const rect = cv.minAreaRect(contour), box = cv.boxPoints(rect);
                  let longest = -1, dx = 0, dy = 0;
                  for (let k = 0; k < 4; k++) {
                    const next = (k + 1) % 4;
                    const x = box[next].x - box[k].x;
                    const y = box[next].y - box[k].y;
                    const length = Math.hypot(x, y);
                    if (length > longest) { longest = length; dx = x / 2; dy = y / 2; }
                  }
                  item.opening_span = [point(rect.center.x - dx, rect.center.y - dy),
                    point(rect.center.x + dx, rect.center.y + dy)];
                }
                elements.push(item);
              } finally { contour.delete(); }
            }
          } finally { region.delete(); contours.delete(); hierarchy.delete(); }
        }
      } finally { binary.delete(); labels.delete(); stats.delete(); centroids.delete(); }
    }
    return {
      schema_version: '1.0',
      image: { id: image.id, width: image.width, height: image.height },
      coordinates: { system: 'normalized', origin: 'top_left', x_axis: 'right', y_axis: 'down',
        normalization: 'x/(width-1), y/(height-1)' },
      meters_per_pixel: null, elements,
      limitations: [
        'Room categories are generic; room function is not predicted.',
        'Connected mask regions can merge adjacent instances.',
        'Physical dimensions and 3D heights require separate calibration.',
        'Browser geometry is traced at a maximum 1024-pixel edge and mapped to normalized image coordinates.',
      ],
    };
  }
  root.FloorplanGeometry = { vectorize };
  if (typeof module === 'object' && module.exports) module.exports = { vectorize };
})(typeof self === 'object' ? self : globalThis);
