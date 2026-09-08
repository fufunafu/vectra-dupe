import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { rangeForSensitivity, heatmapRGB, validateHeatmapData, referencePalette,
  contourInterval, CONTOUR_BANDS, overlayStrength } from '../server/static/heatmap-display.mjs';

assert.equal(rangeForSensitivity(0), 10);
assert.equal(rangeForSensitivity(50), 2.5);
assert.equal(rangeForSensitivity(100), 0.625);
assert.equal(rangeForSensitivity(-100), 10);
assert.equal(rangeForSensitivity(200), 0.625);
assert.equal(overlayStrength(0, 2.5), 0);
assert.equal(overlayStrength(null, 2.5), 0);
assert.equal(overlayStrength(NaN, 2.5), 0);
assert.equal(overlayStrength(1.5, 2.5), 1);
assert.equal(overlayStrength(10, 2.5), 1);
assert.equal(overlayStrength(-0.25, 2.5), overlayStrength(0.25, 2.5));
assert.ok(overlayStrength(0.25, 2.5) < 0.08); // Old linear fade was 0.2.
assert.ok(overlayStrength(0.125, 2.5) < 0.02);
assert.ok(overlayStrength(0.25, 0.625) > overlayStrength(0.25, 2.5));
assert.equal(CONTOUR_BANDS, 7);
assert.equal(contourInterval(2.5), 2.5 / 7);
assert.equal(contourInterval(rangeForSensitivity(100)), 0.625 / 7);
const reference = referencePalette();
assert.equal(reference.length, 256);
assert.deepEqual(reference[0], [1, 1, 0]); // Negative endpoint: yellow.
assert.deepEqual(reference[255], [0, 0.94, 0.94]); // Positive endpoint: cyan.
assert.ok(reference.every(c => c.length === 3 && c.every(v => v >= 0 && v <= 1)));
assert.ok(heatmapRGB(-2.5, 2.5, reference)[0] > 0.99);
assert.ok(heatmapRGB(2.5, 2.5, reference)[0] < 0.01);
const palette = Array.from({ length: 256 }, (_, i) => [i / 255, 0, 1 - i / 255]);
assert.deepEqual(heatmapRGB(null, 2.5, palette), [0.55, 0.55, 0.55]);
assert.deepEqual(heatmapRGB(NaN, 2.5, palette), [0.55, 0.55, 0.55]);
assert.deepEqual(heatmapRGB(-20, 2.5, palette), palette[0]);
assert.deepEqual(heatmapRGB(20, 2.5, palette), palette[255]);
assert.deepEqual(heatmapRGB(0, 2.5, palette), palette[128]);
assert.ok(heatmapRGB(1, 0.625, palette)[0] > heatmapRGB(1, 10, palette)[0]);
const distances = Object.freeze([-4, 0, 4, null]);
const data = { version: 1, mesh_sha256: 'hash', distances, palette };
assert.ok(validateHeatmapData(data, 4, 'hash'));
assert.ok(!validateHeatmapData(data, 3, 'hash'));
assert.ok(!validateHeatmapData(data, 4, 'stale-hash'));
assert.ok(!validateHeatmapData({ ...data, distances: [0, 1, 2, Infinity] }, 4, 'hash'));
assert.ok(!validateHeatmapData({ ...data, palette: [[1, 2, 3]] }, 4, 'hash'));
distances.forEach(d => heatmapRGB(d, 1, palette));
assert.deepEqual(distances, [-4, 0, 4, null]);
const html = readFileSync(new URL('../server/static/index.html', import.meta.url), 'utf8');
const script = html.match(/<script type="module">([\s\S]*?)<\/script>/)[1];
new vm.SourceTextModule(script);
console.log('Heatmap display tests passed, including viewer module syntax.');
