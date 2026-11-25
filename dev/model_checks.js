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
  if (process.env.FLUME_ADVECTION) M.advection = process.env.FLUME_ADVECTION;
  console.log('momentum advection: ' + (M.advection === 'sl' ? 'semi-Lagrangian' : 'conservative finite volume'));
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

  const ONLY = (process.env.FLUME_ONLY || '').split(',').filter(Boolean);
  const want = (name) => !ONLY.length || ONLY.includes(name);
  const dx = 0.02;
  // 1. still water, flat bed of sediment, a submerged solid block and a porous patch
  if (want('still')) {
  let r = flume(10, 1.0, dx, (x, z) => (z < 0.1 ? 4 : (x > 4 && x < 5 && z < 0.3) ? 1 : (x > 6 && x < 7 && z < 0.35) ? 2 : 0));
  var err = await setup(r, { type: 'river', h0: 0.5, q: 0, hout: 0.5 });
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

  }

  // 3. waves: lab flume 20 m, h = 0.4 m, H = 0.06 m, T = 1.6 s
  if (want('waves')) {
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

  }

  // 4. river: q = 0.06 m2/s over a flat bed, h = 0.3 m: mean velocity 0.2 m/s mid-flume
  if (want('river')) {
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

  }

  // ── stage 2: hydraulics against textbook results ──
  const g = 9.81, d1 = 0.01;
  // mean of the surface height at x over the last `secs` seconds
  async function meanEta(xs, secs, dt) {
    const sums = xs.map(() => 0); let n = 0;
    for (let t = 0; t < secs; t += 0.25) {
      await M.debug.steps(Math.round(0.25 / dt), dt);
      const eta = await M.debug.read('eta'); xs.forEach((x, j) => { sums[j] += eta[Math.round(x / M.grid.dx)]; }); n++;
    }
    return sums.map(v => v / n);
  }
  const bedOf = (zb) => (x, z) => (z < zb ? 4 : 0);

  // 5. broad-crested weir, free overflow: specific energy over the crest E = 1.5 h_c (ideal).
  //    A sharp-cornered broad-crested weir (head about 0.1 to 0.3 of the crest length) has a
  //    discharge coefficient near 0.85, so E / 1.5 h_c near (1/0.85)^(2/3) = 1.11.
  if (want('weir')) {
    const zb = 0.02, P = 0.20, q = 0.03, hc = Math.cbrt(q * q / g);
    const rr = flume(8, 0.6, d1, (x, z) => (z < zb ? 4 : (x > 3 && x < 3.4 && z < zb + P) ? 1 : 0));
    await setup(rr, { type: 'river', h0: zb + P + 1.6 * hc, q, hout: 0.05 });
    const dt = stepFor(d1, 1.2); await M.debug.steps(Math.round(35 / dt), dt);
    const [eUp] = await meanEta([2.0], 5, dt);
    const H1 = eUp - zb - P, v = q / (eUp - zb), E = H1 + v * v / (2 * g), ratio = E / (1.5 * hc);
    check('broad-crested weir: energy over the crest 1.03 to 1.20 times 1.5 h_c (discharge coefficient 0.76 to 0.96)', isFinite(ratio) && ratio > 1.03 && ratio < 1.20,
      'E = ' + (E * 1000).toFixed(1) + ' mm, 1.5 h_c = ' + (1.5 * hc * 1000).toFixed(1) + ' mm, ratio ' + ratio.toFixed(3));
  }

  // 6. hydraulic jump below a sluice gate: the depth after the jump matches Belanger's conjugate
  //    depth for the measured depth before it
  if (want('jump')) {
    const zb = 0.02, h1 = 0.04, q = 0.10;
    const Fr = q / (h1 * Math.sqrt(g * h1)), h2 = h1 / 2 * (Math.sqrt(1 + 8 * Fr * Fr) - 1);
    const rr = flume(8, 0.5, d1, (x, z) => (z < zb ? 4 : (x < 0.1 && z > zb + h1) ? 1 : 0));
    await setup(rr, { type: 'river', h0: zb + h1, q, hout: 0.85 * h2 });   // a little under the ideal conjugate, so friction does not drown the jump against the gate
    const dt = stepFor(d1, 3.5); await M.debug.steps(Math.round(30 / dt), dt);
    // time-mean depth profile over 4 s
    const xs = []; for (let x = 0.2; x < 7.8; x += 0.05) xs.push(x);
    const eta = await meanEta(xs, 4, dt), dep = eta.map(e => e - zb);
    const toe = xs.find((x, j) => dep[j] > (h1 + h2) / 2);
    const jt = xs.indexOf(toe);
    // before: the shallowest depth in the metre upstream of the toe; after: mean depth 1 to 2 m past it
    const seg = (a, b) => dep.slice(Math.max(0, jt + Math.round(a / 0.05)), Math.min(dep.length, jt + Math.round(b / 0.05)));
    const before = toe ? Math.min(...seg(-1.0, 0)) : 1, aft = toe ? seg(1.0, 2.0) : [0], after = aft.reduce((x, y) => x + y, 0) / aft.length;
    const FrB = q / (before * Math.sqrt(g * before)), h2B = before / 2 * (Math.sqrt(1 + 8 * FrB * FrB) - 1);
    check('hydraulic jump forms and stays in the flume', toe !== undefined && toe > 0.3 && toe < 6.5, 'toe at ' + (toe ? toe.toFixed(2) : 'none') + ' m');
    check('hydraulic jump: depth after the jump within 12% of the conjugate of the depth before it', toe !== undefined && Math.abs(after - h2B) / h2B < 0.12,
      'before ' + (before * 1000).toFixed(0) + ' mm (Fr ' + FrB.toFixed(2) + '), after ' + (after * 1000).toFixed(0) + ' mm, Belanger ' + (h2B * 1000).toFixed(0) + ' mm');
  }

  // 7. porous block across the whole flume: head drop = integral of (A u + B u^2)/g, superficial u
  if (want('porous')) {
    const zb = 0.02, q = 0.005, n = 0.4, dd = 0.02;
    const rr = flume(8, 0.5, d1, (x, z) => (z < zb ? 4 : (x > 3 && x < 4) ? 2 : 0));
    for (let c = 0; c < rr.mat.length; c++) if (rr.mat[c] === 2) { rr.p1[c] = n; rr.p2[c] = dd; }
    await setup(rr, { type: 'river', h0: zb + 0.30, q, hout: 0.28 });
    const dt = stepFor(d1, 0.3); await M.debug.steps(Math.round(40 / dt), dt);
    const [eu, ed] = await meanEta([2.5, 4.5], 5, dt);
    const hu = eu - zb, hd = ed - zb, dH = (eu - ed) + (q * q / (hu * hu) - q * q / (hd * hd)) / (2 * g);
    const A = 1000 * (1 - n) ** 2 / n ** 3 * 1e-6 / (dd * dd), B = 1.1 * (1 - n) / n ** 3 / dd;
    let th = 0; const nseg = 20; for (let j = 0; j < nseg; j++) { const h = hu + (hd - hu) * (j + 0.5) / nseg, u = q / h; th += (A * u + B * u * u) / g * (1 / nseg); }
    check('porous block: head drop within 15% of Darcy-Forchheimer', th > 0 && isFinite(dH) && Math.abs(dH - th) / th < 0.15, 'drop ' + (dH * 1000).toFixed(1) + ' mm, theory ' + (th * 1000).toFixed(1) + ' mm');
  }

  // 8. plant patch across the whole flume: head drop = integral of 1/2 Cd a u^2 / g over its length
  if (want('plants')) {
    const zb = 0.02, q = 0.03, a = 20, cd = 1.0, Lp = 2;
    const rr = flume(8, 0.5, d1, (x, z) => (z < zb ? 4 : (x > 3 && x < 3 + Lp) ? 3 : 0));
    for (let c = 0; c < rr.mat.length; c++) if (rr.mat[c] === 3) { rr.p1[c] = a; rr.p2[c] = cd; rr.p3[c] = 0.006; }
    await setup(rr, { type: 'river', h0: zb + 0.30, q, hout: 0.28 });
    const dt = stepFor(d1, 0.5); await M.debug.steps(Math.round(40 / dt), dt);
    const [eu, ed] = await meanEta([2.5, 3 + Lp + 0.5], 5, dt);
    const hu = eu - zb, hd = ed - zb, dH = (eu - ed) + (q * q / (hu * hu) - q * q / (hd * hd)) / (2 * g);
    let th = 0; const nseg = 20; for (let j = 0; j < nseg; j++) { const h = hu + (hd - hu) * (j + 0.5) / nseg, u = q / h; th += 0.5 * cd * a * u * u / g * (Lp / nseg); }
    check('plant patch: head drop within 20% of 1/2 Cd a u^2', th > 0 && isFinite(dH) && Math.abs(dH - th) / th < 0.2, 'drop ' + (dH * 1000).toFixed(1) + ' mm, theory ' + (th * 1000).toFixed(1) + ' mm');
    // bed friction shapes the velocity profile upstream (about 10 cm/s): slower next to the bed
    const u = await M.debug.read('u'), i = Math.round(2.0 / d1), kb = Math.round(zb / d1), kt = Math.round((zb + hu) / d1) - 3;
    const ub = u[i + (rr.nx + 1) * kb], us = u[i + (rr.nx + 1) * kt];
    check('bed friction: velocity next to the bed is below the velocity near the surface', us > 0.01 && ub > 0 && ub < 0.95 * us, 'bed ' + (ub * 100).toFixed(2) + ' cm/s, near surface ' + (us * 100).toFixed(2) + ' cm/s');
  }

  // ── stage 3: the wave flume ──
  const WL = require('./wave_lab.js');
  // 9. small waves follow linear dispersion: wavelength from the phase slope over 37 gauges
  if (want('dispersion')) {
    const gauges = []; for (let x = 3; x <= 12.01; x += 0.25) gauges.push(x);
    const m = await WL.measure(M, { T: 1.6, H: 0.01, gauges });
    const Lf = WL.wavelengthFit(m), err = Lf / m.Ltheory - 1;
    check('small waves follow linear dispersion (wavelength within 1%)', isFinite(Lf) && Math.abs(err) < 0.01, Lf.toFixed(3) + ' m, theory ' + m.Ltheory.toFixed(3) + ' m (' + (100 * err).toFixed(2) + '%)');
  }
  // 10. the absorbing far end reflects little (incident/reflected split, three gauges)
  if (want('reflection')) {
    const m = await WL.measure(M, { T: 1.6, H: 0.04 });
    check('the absorbing far end reflects under 8% of the wave', isFinite(m.Kr) && m.Hin > 0.02 && m.Kr < 0.08, 'Kr ' + m.Kr.toFixed(3) + ', incident ' + (m.Hin * 100).toFixed(2) + ' cm');
  }
  // 11. paddle absorption: a wall at the far end, the incident wave must not build up
  if (want('absorption')) {
    const T = 1.6, H = 0.04, h = 0.4, L = 20, nx = L / dx, nz = 0.8 / dx, N = nx * nz;
    const wl = M.debug.pistonStroke(H, T, h), k = 2 * Math.PI / wl.L, om = 2 * Math.PI / T;
    const xs = [6, 6 + 0.22 * wl.L, 6 + 0.5 * wl.L], dtw = 0.4 * dx / 1.0;
    const res = {};
    for (const absorb of [false, true]) {
      const rr = { nx, nz, dx, mat: new Uint8Array(N), p1: new Float32Array(N), p2: new Float32Array(N), p3: new Float32Array(N), cls: new Uint8Array(N) };
      await setup(rr, { type: 'wave', h0: h, H, T, farEnd: 'wall', absorb });
      const inc = [];
      for (const until of [20, 60]) {
        await M.debug.steps(Math.round((until - 6 * T - M.t) / dtw), dtw);
        const every = Math.max(1, Math.round(T / 32 / dtw)), ts = [], ys = xs.map(() => []);
        for (let t = 0; t < 6 * T; t += every * dtw) { await M.debug.steps(every, dtw); ts.push(M.t); const e = await M.debug.read('eta'); xs.forEach((x, j) => ys[j].push(e[Math.round(x / dx)] - h)); }
        inc.push(2 * WL.separate(ys.map(y => WL.harmonic(ts, y, om)), xs, k).aI);
      }
      res[absorb] = inc;
    }
    check('without paddle absorption, waves reflected from a wall build up (the test can see it)', res[false][1] > 1.3 * res[false][0], (res[false][0] * 100).toFixed(2) + ' -> ' + (res[false][1] * 100).toFixed(2) + ' cm');
    check('with paddle absorption the incident wave holds (within 8% from 20 s to 60 s)', Math.abs(res[true][1] / res[true][0] - 1) < 0.08, (res[true][0] * 100).toFixed(2) + ' -> ' + (res[true][1] * 100).toFixed(2) + ' cm');
    M.p.farEnd = 'beach'; M.p.absorb = true;
  }

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
})();
