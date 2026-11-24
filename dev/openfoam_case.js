// Write a test case to a folder: node dev/openfoam_case.js <out dir> <case> [endTime] [dx]
//   cases: weir (broad-crested weir, river), porous (porous block), wave (wave flume over a bar)
'use strict';
const fs = require('fs'), path = require('path'), vm = require('vm');
if (!globalThis.FLUME_CASE) {
  if (!global.registerBackend) global.registerBackend = () => {};
  vm.runInThisContext(fs.readFileSync(path.join(__dirname, '..', 'site.js'), 'utf8') + '\n;globalThis.FLUME_CASE = FLUME_CASE;');
}
const out = process.argv[2], which = process.argv[3] || 'weir', endTime = +(process.argv[4] || 40), dx = +(process.argv[5] || 0.02);
const CASES = {
  weir: { L: 8, H: 0.6, paint: (x, z) => (z < 0.02 ? 4 : (x > 3 && x < 3.4 && z < 0.22) ? 1 : 0), p: { type: 'river', h0: 0.22 + 1.6 * Math.cbrt(0.03 * 0.03 / 9.81), q: 0.03, hout: 0.05 } },
  porous: { L: 8, H: 0.5, paint: (x, z) => (z < 0.02 ? 4 : (x > 3 && x < 4) ? 2 : 0), p1: 0.4, p2: 0.02, p: { type: 'river', h0: 0.32, q: 0.005, hout: 0.28 } },
  wave: { L: 20, H: 0.8, paint: (x, z) => (z < 0.02 ? 4 : (Math.abs(x - 9) < 1.5 - z * 2 && z < 0.28) ? 1 : 0), p: { type: 'wave', h0: 0.42, H: 0.06, T: 1.6 } },
};
const C = CASES[which];
function raster(C, dx) {
  const nx = Math.round(C.L / dx), nz = Math.round(C.H / dx), N = nx * nz;
  const r = { nx, nz, dx, mat: new Uint8Array(N), p1: new Float32Array(N), p2: new Float32Array(N), p3: new Float32Array(N), cls: new Uint8Array(N) };
  for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) { const m = C.paint((i + 0.5) * dx, (k + 0.5) * dx); if (m) { const c = i + nx * k; r.mat[c] = m; if (m === 2) { r.p1[c] = C.p1; r.p2[c] = C.p2; } if (m === 4) r.cls[c] = 1; } }
  return r;
}
module.exports = { CASES, raster };
if (require.main === module) {
  const r = raster(C, dx), L = r.nx * dx;
  const F = FLUME_CASE.files({ grid: { nx: r.nx, nz: r.nz, dx }, raster: r, p: C.p, sx0: L - Math.min(Math.max(2.84, 0.15 * L), 0.4 * L), endTime, scale: 'lab' });
  for (const [name, text] of Object.entries(F)) { const f = path.join(out, name); fs.mkdirSync(path.dirname(f), { recursive: true }); fs.writeFileSync(f, text); }
  fs.chmodSync(path.join(out, 'Allrun'), 0o755);
  console.log('wrote', Object.keys(F).length, 'files to', out, '(' + r.nx + ' x ' + r.nz + ')');
}
