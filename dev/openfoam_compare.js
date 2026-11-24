// Compare the browser model with an OpenFOAM run of the exported case.
//   node dev/openfoam_compare.js <case: weir|porous|wave> <OpenFOAM case dir> [endTime] [dx]
// Runs model.js (Dawn, needs a GPU) on the same raster as dev/openfoam_case.js, and compares
// the time-mean water surface at the case's 19 probe points over the last 5 s (for the wave
// case: wave height, max minus min, over the last 3 periods).
'use strict';
const fs = require('fs'), path = require('path'), vm = require('vm');
const { CASES, raster } = require('./openfoam_case.js');

(async () => {
  const which = process.argv[2], dir = process.argv[3], endTime = +(process.argv[4] || 40), dx = +(process.argv[5] || 0.02);
  const C = CASES[which], r = raster(C, dx), L = r.nx * dx;
  const xs = []; for (let j = 1; j < 20; j++) xs.push(L * j / 20);
  const wave = C.p.type === 'wave', win = wave ? 3 * C.p.T : 5;

  // OpenFOAM: hL (elevation of the surface) per probe, time-mean (or range) over the window
  const lines = fs.readFileSync(path.join(dir, 'postProcessing/surface/0/height.dat'), 'utf8').split('\n').filter(l => l && l[0] !== '#');
  const rows = lines.map(l => l.trim().split(/\s+/).map(Number)).filter(a => a[0] >= endTime - win - 1e-9);
  const of = xs.map((_, j) => { const v = rows.map(a => a[2 + 2 * j]); return wave ? Math.max(...v) - Math.min(...v) : v.reduce((s, x) => s + x, 0) / v.length; });
  const ofEnd = rows.length ? rows[rows.length - 1][0] : 0;

  // browser model
  const { create, globals } = await import('webgpu'); Object.assign(globalThis, globals);
  const opts = ['backend=vulkan']; if (process.env.FLUME_ADAPTER) opts.unshift('adapter=' + process.env.FLUME_ADAPTER);
  let M = null; globalThis.registerModel = (m) => { M = m; };
  vm.runInThisContext(fs.readFileSync(path.join(__dirname, '..', 'model.js'), 'utf8'), { filename: 'model.js' });
  const res = await M.init({ gpu: create(opts) }); if (!res.ok) { console.log(res.reason); process.exit(1); }
  if (process.env.FLUME_KCG) M.kcg = +process.env.FLUME_KCG;
  M.setScale('lab'); Object.assign(M.p, C.p); M.setGrid({ nx: r.nx, nz: r.nz, dx }, r); M.reset();
  const dt = 0.4 * dx / 1.5;
  await M.debug.steps(Math.round((endTime - win) / dt), dt);
  const acc = xs.map(() => (wave ? [1e9, -1e9] : 0)); let n = 0;
  const every = Math.max(1, Math.round((wave ? C.p.T / 24 : 0.25) / dt));
  for (let t = 0; t < win; t += every * dt) {
    await M.debug.steps(every, dt);
    const eta = await M.debug.read('eta');
    xs.forEach((x, j) => { const e = eta[Math.min(r.nx - 1, Math.round(x / dx))]; if (wave) { acc[j][0] = Math.min(acc[j][0], e); acc[j][1] = Math.max(acc[j][1], e); } else acc[j] += e; });
    n++;
  }
  const fl = acc.map(a => (wave ? a[1] - a[0] : a / n));

  console.log(`${which}: ${r.nx} x ${r.nz} cells of ${dx * 100} cm, ${wave ? 'wave height' : 'mean surface'} over the last ${win.toFixed(1)} s (OpenFOAM reached ${ofEnd} s)`);
  console.log('   x (m)   OpenFOAM    browser    diff (mm)');
  let se = 0, cnt = 0;
  xs.forEach((x, j) => {
    if (!(of[j] > 0) || !(fl[j] > 0)) { console.log(`  ${x.toFixed(2).padStart(6)}   ${'-'.padStart(8)}   ${'-'.padStart(8)}`); return; }
    const d = (fl[j] - of[j]) * 1000; se += d * d; cnt++;
    console.log(`  ${x.toFixed(2).padStart(6)}   ${(of[j] * 1000).toFixed(1).padStart(8)}   ${(fl[j] * 1000).toFixed(1).padStart(8)}   ${d.toFixed(1).padStart(8)}`);
  });
  console.log(`  RMS difference ${Math.sqrt(se / Math.max(cnt, 1)).toFixed(1)} mm over ${cnt} probes`);
  if (cnt < 10 || ofEnd < endTime - 1e-6) { console.log(`  NOT A COMPARISON: OpenFOAM reached ${ofEnd} s of ${endTime} s and ${cnt} probes have data`); process.exit(2); }
  if (which === 'weir') {
    const g = 9.81, q = C.p.q, hc = Math.cbrt(q * q / g), j = 4;   // probe at x = 2.0 m
    const E = (e) => (e - 0.22) + (q / (e - 0.02)) ** 2 / (2 * g);
    console.log(`  energy over the crest / 1.5 h_c at x = ${xs[j].toFixed(1)} m: OpenFOAM ${(E(of[j]) / (1.5 * hc)).toFixed(3)}, browser ${(E(fl[j]) / (1.5 * hc)).toFixed(3)}`);
  }
  if (which === 'porous') {
    const ju = xs.findIndex(x => x >= 2.4), jd = xs.findIndex(x => x >= 4.4);
    console.log(`  head drop across the block (x ${xs[ju].toFixed(1)} to ${xs[jd].toFixed(1)} m): OpenFOAM ${((of[ju] - of[jd]) * 1000).toFixed(1)} mm, browser ${((fl[ju] - fl[jd]) * 1000).toFixed(1)} mm`);
  }
  process.exit(0);
})();
