// Display controls only. These helpers never modify distances or measurements.
export const CONTOUR_BANDS = 7;
export const OVERLAY_FULL_TINT_FRACTION = 0.6;

// A display fade, not a detection threshold. Suppress the near-zero white wash.
export function overlayStrength(distance, range) {
  if (distance === null || !Number.isFinite(distance)) return 0;
  const t = Math.min(1, Math.abs(distance) / (range * OVERLAY_FULL_TINT_FRACTION));
  return t * t * (3 - 2 * t);
}

export function contourInterval(range) { return range / CONTOUR_BANDS; }

export function referencePalette() {
  // Reference display: negative yellow/orange, neutral white, positive blue/cyan.
  // Only the color convention changes. Signed distances are never inverted.
  const stops = [
    [0, [1, 1, 0]], [0.14, [1, 0.67, 0.12]], [0.29, [0.94, 0.56, 0.40]],
    [0.43, [0.98, 0.85, 0.77]], [0.5, [0.97, 0.97, 0.97]],
    [0.60, [0.76, 0.89, 0.97]], [0.73, [0.32, 0.66, 0.92]],
    [0.87, [0, 0.61, 0.94]], [1, [0, 0.94, 0.94]],
  ];
  return Array.from({ length: 256 }, (_, i) => {
    const t = i / 255;
    const right = stops.findIndex(s => s[0] >= t);
    if (right === 0) return [...stops[0][1]];
    const [p, a] = stops[right - 1], [q, b] = stops[right];
    const fraction = (t - p) / (q - p);
    return a.map((v, k) => v + (b[k] - v) * fraction);
  });
}

export function rangeForSensitivity(value) {
  const sensitivity = Math.max(0, Math.min(100, Number(value)));
  // 0 = +/-10, 50 = +/-2.5, 100 = +/-0.625. Right reveals smaller changes.
  return 10 * Math.pow(16, -sensitivity / 100);
}

export function heatmapRGB(distance, range, palette) {
  if (distance === null || !Number.isFinite(distance)) return [0.55, 0.55, 0.55];
  const normalized = Math.max(0, Math.min(1, (distance + range) / (2 * range)));
  return palette[Math.min(palette.length - 1, Math.floor(normalized * palette.length))];
}

export function validateHeatmapData(data, vertexCount, meshHash) {
  return data?.version === 1 && data.mesh_sha256 === meshHash &&
    Array.isArray(data.distances) && data.distances.length === vertexCount &&
    data.distances.every(d => d === null || Number.isFinite(d)) &&
    Array.isArray(data.palette) && data.palette.length === 256 &&
    data.palette.every(c => Array.isArray(c) && c.length === 3 &&
      c.every(v => Number.isFinite(v) && v >= 0 && v <= 1));
}
