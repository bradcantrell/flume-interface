// Checks for the flume model (model.js), on a real GPU through Dawn (npm package `webgpu`).
//   node dev/model_checks.js            (FLUME_ADAPTER=<part of the GPU name> picks one GPU when there are several)
// Stage 1 checks: shaders build, still water stays still, walls hold water out, the paddle
// makes waves near linear theory and the sponge takes them out, the river carries its discharge.
// Tighter accuracy checks (analytic + OpenFOAM) come with stages 2 and 3.
'use strict';
const fs = require('fs'), path = require('path'), vm = require('vm');

(async () => {
  const { create, globals } = await import('webgpu');
  Object.assign(globalThis, globals);
  const opts = ['backend=vulkan']; if (process.env.FLUME_ADAPTER) opts.unshift('adapter=' + process.env.FLUME_ADAPTER);
  const gpu = create(opts);
  let M = null;
  globalThis.registerModel = (m) => { M = m; };
  vm.runInThisContext(fs.readFileSync(path.join(__dirname, '..', 'model.js'), 'utf8'), { filename: 'model.js' });

  let pass = 0, fail = 0;
  const check = (name, ok, detail) => { if (ok) { pass++; console.log('  ok  ' + name + (detail ? '  (' + detail + ')' : '')); } else { fail++; console.log('  FAIL ' + name + (detail ? '  — ' + detail : '')); } };
  console.log('flume model checks');

  const res = await M.init({ gpu });
  check('init finds a GPU', res.ok, res.reason);
  if (!res.ok) process.exit(1);

  function flume(L, H, dx, paint) {
    const nx = Math.round(L / dx), nz = Math.round(H / dx), N = nx * nz;
    const r = { nx, nz, dx, mat: new Uint8Array(N), p1: new Float32Array(N), p2: new Float32Array(N), p3: new Float32Array(N), cls: new Uint8Array(N) };
    if (paint) for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) { const m = paint((i + 0.5) * dx, (k + 0.5) * dx); if (m) r.mat[i + nx * k] = m; }
    return r;
  }
  async function setup(r, params) {
    M.device.pushErrorScope('validation');
    M.setScale('lab'); Object.assign(M.p, params);
    M.setGrid({ nx: r.nx, nz: r.nz, dx: r.dx }, r); M.reset();
    const err = await M.device.popErrorScope();
    return err ? err.message : null;
  }
  // top speed on faces next to water (air faces carry extrapolated values only)
  async function maxSpeed(r) {
    const u = await M.debug.read('u'), w = await M.debug.read('w'), f = await M.debug.read('f'), nx = r.nx, nz = r.nz; let m = 0;
    const wet = (i, k) => i >= 0 && i < nx && k >= 0 && k < nz && f[i + nx * k] > 0.5;
    for (let k = 0; k < nz; k++) for (let i = 0; i <= nx; i++) if (wet(i - 1, k) || wet(i, k)) m = Math.max(m, Math.abs(u[i + (nx + 1) * k]));
    for (let k = 0; k <= nz; k++) for (let i = 0; i < nx; i++) if (wet(i, k - 1) || wet(i, k)) m = Math.max(m, Math.abs(w[i + nx * k]));
    return m;
  }
  async function volume(r) { const f = await M.debug.read('f'); let s = 0; for (let c = 0; c < r.nx * r.nz; c++) s += f[c]; return s * r.dx * r.dx; }
  const stepFor = (dx, v) => 0.4 * dx / v;

  // 1. still water, flat bed of sediment, a submerged solid block and a porous patch
  const dx = 0.02;
  let r = flume(10, 1.0, dx, (x, z) => (z < 0.1 ? 4 : (x > 4 && x < 5 && z < 0.3) ? 1 : (x > 6 && x < 7 && z < 0.35) ? 2 : 0));
  let err = await setup(r, { type: 'river', h0: 0.5, q: 0, hout: 0.5 });
  check('every pipeline builds and binds without a validation error', !err, err);
  const rerr = await M.debug.validateRender(400, 200);
  check('the water renderer builds and draws', !rerr, rerr);
  const v0 = await volume(r);
  await M.debug.steps(400, stepFor(dx, 1.0));
  const vs = await maxSpeed(r), v1 = await volume(r);
  check('still water stays still (top speed under 5 mm/s after 400 steps)', vs < 0.005, (vs * 1000).toFixed(2) + ' mm/s');
  check('still water keeps its volume (within 0.2%)', Math.abs(v1 - v0) / v0 < 0.002, (100 * (v1 - v0) / v0).toFixed(3) + '%');
  const f = await M.debug.read('f'); let wet = 0;
  for (let c = 0; c < r.nx * r.nz; c++) if ((r.mat[c] === 1 || r.mat[c] === 4) && f[c] > 0) wet++;
  check('no water inside solid or sediment cells', wet === 0, wet + ' cells');

  // 2. a wall drawn while water is there pushes it out cleanly
  const r2 = flume(10, 1.0, dx, (x, z) => (z < 0.1 ? 4 : (x > 2 && x < 2.6 && z < 0.8) ? 1 : 0));
  M.setRaster(r2); await M.debug.steps(5, stepFor(dx, 1.0));
  const f2 = await M.debug.read('f'); let wet2 = 0; for (let c = 0; c < r2.nx * r2.nz; c++) if (r2.mat[c] === 1 && f2[c] > 0) wet2++;
  check('drawing a wall into the water empties its cells', wet2 === 0, wet2 + ' cells');

  // 3. waves: lab flume 20 m, h = 0.4 m, H = 0.06 m, T = 1.6 s
  const rw = flume(20, 0.8, dx, () => 0);
  err = await setup(rw, { type: 'wave', h0: 0.4, H: 0.06, T: 1.6 });
  const wt = M.debug.pistonStroke(0.06, 1.6, 0.4);
  const xs = [wt.L * 1.0, wt.L * 1.5, wt.L * 2.0], iS = xs.map(x => Math.round(x / dx));
  const iEnd = Math.round((20 - 0.25 * (20 - M.P.sx0)) / dx);   // inside the sponge
  const dtw = stepFor(dx, 1.0), T = 1.6, settle = 7 * T, measure = 3 * T;
  await M.debug.steps(Math.round(settle / dtw), dtw);
  const hi = xs.map(() => -1e9), lo = xs.map(() => 1e9); let hiE = -1e9, loE = 1e9;
  const every = Math.max(1, Math.round(T / 24 / dtw));
  for (let t = 0; t < measure; t += every * dtw) {
    await M.debug.steps(every, dtw);
    const eta = await M.debug.read('eta');
    iS.forEach((i, j) => { hi[j] = Math.max(hi[j], eta[i]); lo[j] = Math.min(lo[j], eta[i]); });
    hiE = Math.max(hiE, eta[iEnd]); loE = Math.min(loE, eta[iEnd]);
  }
  const Hs = hi.map((h, j) => h - lo[j]), Hmean = Hs.reduce((a, b) => a + b, 0) / Hs.length;
  check('the paddle makes waves of the asked height (within 12%, linear wavemaker)', Hs.every(h => Math.abs(h - 0.06) / 0.06 < 0.12),
    'H at 1, 1.5, 2 wavelengths: ' + Hs.map(h => (h * 100).toFixed(1)).join(', ') + ' cm; asked 6.0 cm; wavelength ' + wt.L.toFixed(2) + ' m');
  check('the sponge takes the waves out at the far end (under a fifth of the height)', (hiE - loE) < Hmean / 5, ((hiE - loE) * 100).toFixed(1) + ' cm');

  // 4. river: q = 0.06 m2/s over a flat bed, h = 0.3 m: mean velocity 0.2 m/s mid-flume
  const rr = flume(10, 0.8, dx, (x, z) => (z < 0.04 ? 4 : 0));
  err = await setup(rr, { type: 'river', h0: 0.34, q: 0.06, hout: 0.30 });   // water level 0.34 over a 0.04 bed: depth 0.30
  const dtr = stepFor(dx, 0.6);
  await M.debug.steps(Math.round(30 / dtr), dtr);
  // discharge through the middle, averaged over 10 s (the flume still sloshes: no friction until stage 2)
  const im = Math.round(5 / dx), vA = await volume(rr); let qsum = 0, nq = 0;
  for (let t = 0; t < 10; t += 0.25) {
    await M.debug.steps(Math.round(0.25 / dtr), dtr);
    const u = await M.debug.read('u'), ff = await M.debug.read('f');
    let qx = 0; for (let k = 0; k < rr.nz; k++) qx += u[im + (rr.nx + 1) * k] * Math.min(1, Math.max(0, ff[im + rr.nx * k])) * dx;
    qsum += qx; nq++;
  }
  const vB = await volume(rr), qm = qsum / nq;
  check('the river carries its discharge through the middle (10 s mean, within 3%)', Math.abs(qm - 0.06) / 0.06 < 0.03, 'q = ' + qm.toFixed(4) + ' m²/s per m, asked 0.06');
  check('the river holds its volume over 10 s (within 0.5%)', Math.abs(vB - vA) / vA < 0.005, (100 * (vB - vA) / vA).toFixed(2) + '%');

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
})();
