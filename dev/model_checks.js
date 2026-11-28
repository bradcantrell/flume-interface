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
    M.setScale('lab'); Object.assign(M.p, { sediment: false }, params);   // the moving bed is off unless a check turns it on
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
    // the measured section: the energy lost across the jump, as marked on the energy line, against
    // the classical loss for the measured depths, (y2 - y1)^3 / (4 y1 y2)
    const sdt = dt; for (let tt = 0; tt < 10; tt += 0.25) { await M.debug.steps(Math.round(0.25 / sdt), sdt); await M.debug.readColumns(); }
    const sec = M.sectionLines(), mk = sec && sec.marks.length ? sec.marks.slice().sort((a, b) => (b.z0 - b.z1) - (a.z0 - a.z1))[0] : null;
    // two textbook estimates of the loss for the measured time-mean depths before (y1) and after
    // (y2): the energy balance E = z + y + q^2/(2 g y^2), and the classical (y2 - y1)^3 / (4 y1 y2),
    // which assumes y2 is exactly conjugate; a real flume sits between them
    const Eb = zb + before + q * q / (2 * g * before * before), Ea = zb + after + q * q / (2 * g * after * after);
    const dBal = Eb - Ea, dCls = toe ? Math.pow(after - before, 3) / (4 * before * after) : 0, dE = mk ? mk.z0 - mk.z1 : 0;
    const lo = 0.9 * Math.min(dBal, dCls), hi = 1.1 * Math.max(dBal, dCls);
    check('measured section: the energy loss marked at the jump lies between the two textbook estimates (within 10%), at the jump', mk && dE > lo && dE < hi && toe > mk.x0 - 0.3 && toe < mk.x1 && mk.x1 - mk.x0 < 2.0,
      (process.env.FLUME_DEBUG ? JSON.stringify(sec.marks.map(m => [m.x0.toFixed(2), m.x1.toFixed(2), ((m.z0 - m.z1) * 1000).toFixed(1)])) + ' ' : '') + 'marked ' + (dE * 1000).toFixed(1) + ' mm over ' + (mk ? mk.x0.toFixed(2) + '-' + mk.x1.toFixed(2) + ' m' : '-') + ', energy balance ' + (dBal * 1000).toFixed(1) + ' mm, classical ' + (dCls * 1000).toFixed(1) + ' mm, toe ' + (toe ? toe.toFixed(2) : '-') + ' m');
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

  // ── stage 4: sediment ──
  const readBed = async () => { const b = await M.debug.read('bed'), n = b.length / 3, zs = new Float32Array(n), zf = new Float32Array(n); for (let i = 0; i < n; i++) { zs[i] = b[3 * i]; zf[i] = b[3 * i + 1]; } return { zs, zf }; };
  const bedVolume = (b) => { let v = 0; for (let i = 0; i < b.zs.length; i++) if (b.zs[i] >= 0) v += b.zs[i] - b.zf[i]; return v; };
  const sandFlume = (L, Hd, d, zb, cls, extra) => flume(L, Hd, d, (x, z) => (z < 0.02 ? 1 : z < zb(x) ? 4 : (extra ? extra(x, z) : 0)));
  const setCls = (r, cls) => { for (let c = 0; c < r.mat.length; c++) if (r.mat[c] === 4) r.cls[c] = cls; return r; };
  // 12. still water: the bed does not move
  if (want('sedstill')) {
    const rr = setCls(sandFlume(6, 0.6, dx, () => 0.12), 0);
    await setup(rr, { type: 'river', h0: 0.40, q: 0, hout: 0.38, sediment: true, morfac: 20 });
    const b0 = await readBed(); await M.debug.steps(600, stepFor(dx, 1.0)); const b1 = await readBed();
    let mx = 0; for (let i = 0; i < b0.zs.length; i++) mx = Math.max(mx, Math.abs(b1.zs[i] - b0.zs[i]));
    check('still water: the bed does not move', mx < 1e-5, 'largest change ' + (mx * 1000).toFixed(3) + ' mm');
  }
  // 13. the bedload the GPU computes is Meyer-Peter & Mueller for the bed shear it computed
  if (want('sedmpm')) {
    const rr = setCls(sandFlume(8, 0.6, dx, () => 0.10), 0);
    await setup(rr, { type: 'river', h0: 0.36, q: 0.12, hout: 0.26, sediment: true, morfac: 1 });
    await M.debug.steps(Math.round(15 / stepFor(dx, 1.0)), stepFor(dx, 1.0));
    const ust = await M.debug.read('ust'), qc = await M.debug.read('qc'), gp = M.debug.grainProps(0.0002), R = 1.65;
    let worst = 0, nmov = 0;
    for (let i = 100; i < 300; i += 10) {
      const th = ust[i] * ust[i] / (R * 9.81 * 0.0002), q = 8 * Math.pow(Math.max(th - gp.thc, 0), 1.5) * Math.sqrt(R * 9.81 * 0.0002 ** 3);
      if (q > 0) { nmov++; worst = Math.max(worst, Math.abs(Math.abs(qc[3 * i]) - q) / q); }
    }
    check('bedload rate is Meyer-Peter & Mueller for the bed shear (fine sand, within 1%)', nmov >= 10 && worst < 0.01, nmov + ' columns moving, worst ' + (100 * worst).toFixed(3) + '%, critical Shields ' + gp.thc.toFixed(4));
  }
  // 14. a closed wave flume keeps its bedload sediment (coarse sand, which stays on the bed)
  if (want('sedmass')) {
    const rr = setCls(sandFlume(12, 0.8, dx, (x) => 0.08 + 0.025 * x), 1);
    await setup(rr, { type: 'wave', h0: 0.42, H: 0.14, T: 2.0, farEnd: 'wall', absorb: true, sediment: true, morfac: 20 });
    const b0 = await readBed(); await M.debug.steps(Math.round(30 / stepFor(dx, 1.0)), stepFor(dx, 1.0)); const b1 = await readBed();
    const v0 = bedVolume(b0), v1 = bedVolume(b1); let moved = 0; for (let i = 0; i < b0.zs.length; i++) moved = Math.max(moved, Math.abs(b1.zs[i] - b0.zs[i]));
    check('a closed wave flume keeps its sediment (bed volume within 0.1%)', moved > 0.002 && Math.abs(v1 - v0) / v0 < 0.001, 'volume ' + (100 * (v1 - v0) / v0).toFixed(4) + '%, largest bed change ' + (moved * 100).toFixed(1) + ' cm');
  }
  // 15. a 45 degree mound slumps to the angle of repose (32 degrees) and keeps its volume
  if (want('repose')) {
    const rr = setCls(sandFlume(6, 0.8, dx, (x) => 0.08 + Math.max(0, 0.4 - Math.abs(x - 3))), 1);
    await setup(rr, { type: 'river', h0: 0.70, q: 0, hout: 0.68, sediment: true, morfac: 1 });
    const b0 = await readBed(); await M.debug.steps(300, stepFor(dx, 1.0)); const b1 = await readBed();
    let smax = 0; for (let i = 1; i < b1.zs.length; i++) if (b1.zs[i] >= 0 && b1.zs[i - 1] >= 0) smax = Math.max(smax, Math.abs(b1.zs[i] - b1.zs[i - 1]) / dx);
    const v0 = bedVolume(b0), v1 = bedVolume(b1), tanp = Math.tan(32 * Math.PI / 180);
    check('a 45 degree mound slumps to the angle of repose and keeps its volume', smax <= tanp + 0.01 && Math.abs(v1 - v0) / v0 < 1e-4, 'steepest ' + (Math.atan(smax) * 180 / Math.PI).toFixed(1) + ' degrees (repose 32), volume ' + (100 * (v1 - v0) / v0).toFixed(5) + '%');
  }
  // 16. a strong river over a sill digs a scour hole downstream of it
  if (want('scour')) {
    const rr = setCls(sandFlume(10, 0.6, dx, () => 0.10, 1, (x, z) => (x > 4 && x < 4.3 && z < 0.20 ? 1 : 0)), 1);
    await setup(rr, { type: 'river', h0: 0.36, q: 0.15, hout: 0.22, sediment: true, morfac: 30 });
    const b0 = await readBed(); await M.debug.steps(Math.round(20 / stepFor(dx, 1.5)), stepFor(dx, 1.5)); const b1 = await readBed();
    let lo = 0, xlo = 0; for (let i = Math.round(4.3 / dx); i < Math.round(6.5 / dx); i++) { const d = b1.zs[i] - b0.zs[i]; if (d < lo) { lo = d; xlo = i * dx; } }
    let up = 0; for (let i = Math.round(1 / dx); i < Math.round(3 / dx); i++) up = Math.min(up, b1.zs[i] - b0.zs[i]);
    check('a river over a sill scours the bed just downstream of it', lo < -0.01 && xlo < 5.5 && lo < 2 * up, 'deepest ' + (lo * 100).toFixed(1) + ' cm at x = ' + xlo.toFixed(2) + ' m (sill ends at 4.3 m), upstream reach ' + (up * 100).toFixed(1) + ' cm');
  }

  // 17. settling column: fine sand starts mixed through still water over a gravel bed; half has
  //     settled out at t = h / (2 ws)
  if (want('settling')) {
    const rr = setCls(sandFlume(2, 0.6, dx, () => 0.10), 2);
    await setup(rr, { type: 'river', h0: 0.50, q: 0, hout: 0.48, sediment: true, morfac: 1 });
    const N = rr.nx * rr.nz, f0 = await M.debug.read('f'), cs = new Float32Array(3 * N), C0 = 0.001;
    for (let c = 0; c < N; c++) if (f0[c] > 0.999 && rr.mat[c] === 0) cs[3 * c] = C0;
    M.debug.write('cs', cs);
    const mass = async () => { const c2 = await M.debug.read('cs'); let m = 0; for (let c = 0; c < N; c++) m += c2[3 * c]; return m; };
    // the depth of the sand column: the filled cells in the middle column
    let nFull = 0; const im = Math.round(rr.nx / 2); for (let k = 0; k < rr.nz; k++) if (cs[3 * (im + rr.nx * k)] > 0) nFull++;
    const m0 = await mass(), ws = M.debug.grainProps(0.0002).ws, h = nFull * dx, tHalf = h / (2 * ws), dtS = 0.004;
    await M.debug.steps(Math.round(tHalf / dtS), dtS);
    const ratio = (await mass()) / m0;
    check('settling column: half the fine sand has settled out at t = h / (2 ws) (within 5%)', Math.abs(ratio - 0.5) < 0.025, 'remaining ' + (100 * ratio).toFixed(1) + '% at ' + tHalf.toFixed(2) + ' s, fall velocity ' + (ws * 100).toFixed(2) + ' cm/s');
  }
  // 18. a steady river: the time-mean suspended profile follows Rouse for the model's own mean u*.
  //     At 2 cm cells the scheme's own discretization error is 10 to 20% (a 1D replica of the scheme
  //     shows the same and converges as cells shrink), so the check allows 25%.
  if (want('rouse')) {
    const rr = setCls(sandFlume(14, 0.6, dx, () => 0.10), 0);
    await setup(rr, { type: 'river', h0: 0.40, q: 0.20, hout: 0.30, sediment: true, morfac: 1 });
    const dtR = stepFor(dx, 1.5), i = Math.round(11 / dx); await M.debug.steps(Math.round(40 / dtR), dtR);
    const Cm = new Float64Array(rr.nz); let usS = 0, hS = 0, n = 0;
    for (let t = 0; t < 10; t += 0.25) { await M.debug.steps(Math.round(0.25 / dtR), dtR); const cs = await M.debug.read('cs'), u = await M.debug.read('ust'), e = await M.debug.read('eta'); for (let k = 0; k < rr.nz; k++) Cm[k] += cs[3 * (i + rr.nx * k)]; usS += u[i]; hS += e[i]; n++; }
    const b = await readBed(), zb = b.zs[i], h = hS / n - zb, P = M.debug.grainProps(0.0002).ws / (0.41 * usS / n);
    const rz = (k) => (k + 0.5) * dx - zb, kOf = (z) => Math.floor((zb + z) / dx), ka = kOf(0.15 * h), zA = rz(ka), rows = [];
    let worst = 0;
    for (const fz of [0.3, 0.5, 0.7]) {
      const k = kOf(fz * h), zz = rz(k), rouse = Math.pow(((h - zz) / zz) * (zA / (h - zA)), P), model = Cm[k] / Cm[ka];
      worst = Math.max(worst, Math.abs(model / rouse - 1)); rows.push(fz + 'h ' + model.toFixed(3) + ' vs ' + rouse.toFixed(3));
    }
    check('time-mean suspended profile follows Rouse for the model\'s u* (within 25% at 0.3, 0.5, 0.7 of the depth)', isFinite(worst) && Cm[ka] > 0 && worst < 0.25, 'Rouse number ' + P.toFixed(2) + '; ' + rows.join(', '));
  }
  // 19. bed plus suspended sediment is conserved in a closed wave flume (bed speed-up 1)
  if (want('susmass')) {
    const rr = setCls(sandFlume(12, 0.8, dx, (x) => 0.08 + 0.025 * x), 0);
    await setup(rr, { type: 'wave', h0: 0.42, H: 0.12, T: 2.0, farEnd: 'wall', absorb: true, sediment: true, morfac: 1 });
    const N = rr.nx * rr.nz, total = async () => { const b = await readBed(), c2 = await M.debug.read('cs'); let m = 0; for (let c = 0; c < N; c++) m += c2[3 * c] + c2[3 * c + 1] + c2[3 * c + 2]; return { bed: bedVolume(b) * dx * 0.6, sus: m * dx * dx }; };
    const t0 = await total(); await M.debug.steps(Math.round(30 / stepFor(dx, 1.0)), stepFor(dx, 1.0)); const t1 = await total();
    const tot0 = t0.bed + t0.sus, tot1 = t1.bed + t1.sus;
    check('bed plus suspended sediment is conserved in a closed wave flume (within 0.1%)', t1.sus > 0 && Math.abs(tot1 - tot0) / tot0 < 0.001, 'suspended ' + (t1.sus * 1e6).toFixed(1) + ' cm2, total change ' + (100 * (tot1 - tot0) / tot0).toFixed(4) + '%');
  }

  // 20. armouring: clear water (no sediment supply) over a 50/50 bed of fine sand and gravel, a flow
  //     that moves sand but not gravel: the degrading bed ends with a surface richer in gravel
  if (want('armour')) {
    const rr = sandFlume(10, 0.6, dx, () => 0.12);
    setCls(rr, 0);                              // painted as one class; every bed cell is set to a 50/50 mix below
    await setup(rr, { type: 'river', h0: 0.40, q: 0.12, hout: 0.28, sediment: true, morfac: 30, feed: false });
    // start every bed cell as a true 50/50 mix
    const N = rr.nx * rr.nz, cell = await M.debug.read('cell');
    for (let c = 0; c < N; c++) if (rr.mat[c] === 4) { cell[4 * c] = 0.5; cell[4 * c + 1] = 0; cell[4 * c + 2] = 0.5; }
    M.debug.write('cell', cell);
    await M.debug.steps(Math.round(20 / stepFor(dx, 1.0)), stepFor(dx, 1.0));
    const c2 = await M.debug.read('cell'), b = await readBed();
    let g = 0, n = 0;
    for (let i = Math.round(0.5 / dx); i < Math.round(3.5 / dx); i++) { const kt = Math.max(0, Math.ceil(b.zs[i] / dx - 0.5) - 1), ct = i + rr.nx * kt, t = c2[4 * ct] + c2[4 * ct + 1] + c2[4 * ct + 2]; if (t > 0) { g += c2[4 * ct + 2] / t; n++; } }
    const gm = g / n;
    check('armouring: fine sand leaves a mixed bed and the surface coarsens (gravel above 60%)', n > 50 && gm > 0.6, 'surface gravel fraction ' + (100 * gm).toFixed(1) + '% (started at 50%)');
  }

  // 21. flexible plants: the GPU's reconfiguration is Luhar & Nepf for its own canopy velocity, and a
  //     flexible meadow resists less than the same meadow rigid
  if (want('plantflex')) {
    const zb = 0.02, q = 0.06, a = 20, cd = 1.0, Lp = 2, stem = 0.006, drops = {};
    let rGPU = 0, rLN = 0, Ucan = 0;
    for (const EI of [1e6, 2e-5]) {
      const rr = flume(8, 0.5, dx, (x, z) => (z < zb ? 4 : (x > 3 && x < 3 + Lp && z < 0.22) ? 3 : 0));
      for (let c = 0; c < rr.mat.length; c++) if (rr.mat[c] === 3) { rr.p1[c] = a; rr.p2[c] = cd; rr.p3[c] = stem; rr.p4 = rr.p4 || new Float32Array(rr.mat.length); rr.p4[c] = EI; }
      await setup(rr, { type: 'river', h0: zb + 0.30, q, hout: 0.28 });
      const dtp = stepFor(dx, 0.5); await M.debug.steps(Math.round(40 / dtp), dtp);
      const [eu, ed] = await meanEta([2.5, 3 + Lp + 0.5], 5, dtp);
      drops[EI] = eu - ed;
      if (EI < 1) {
        const po = await M.debug.read('po'), u = await M.debug.read('u'), i = Math.round(4 / dx), l = 0.20;
        let us = 0, n = 0; for (let k = Math.floor(zb / dx); k < Math.ceil((zb + l) / dx); k++) { us += 0.5 * (u[i + (rr.nx + 1) * k] + u[i + 1 + (rr.nx + 1) * k]); n++; }
        Ucan = us / n; const Ca = 1000 * cd * stem * Ucan * Ucan * l ** 3 / EI;
        rLN = Math.min(1, Math.max(0.05, 1 - (1 - 0.9 * Math.pow(Ca, -1 / 3)) / (1 + 8 * Math.pow(Ca, -1.5)))); rGPU = po[2 * i];
      }
    }
    check('flexible plants bend as Luhar & Nepf for the canopy velocity (within 2%)', rGPU > 0 && Math.abs(rGPU / rLN - 1) < 0.02, 'l_e/l ' + rGPU.toFixed(3) + ' vs ' + rLN.toFixed(3) + ' at ' + (Ucan * 100).toFixed(1) + ' cm/s');
    check('a flexible meadow resists less than the same meadow rigid', drops[2e-5] > 0 && drops[2e-5] < 0.9 * drops[1e6], 'head drop ' + (drops[2e-5] * 1000).toFixed(1) + ' mm flexible vs ' + (drops[1e6] * 1000).toFixed(1) + ' mm rigid, bent to ' + (100 * rGPU).toFixed(0) + '% of its length');
  }

  // 22. dye: released in a closed wave flume, carried by the waves, and its total kept
  if (want('dye')) {
    const rr = flume(10, 0.8, dx, (x, z) => (z < 0.04 ? 1 : 0));
    await setup(rr, { type: 'wave', h0: 0.42, H: 0.08, T: 1.8, farEnd: 'wall', absorb: true });
    M.injectDye(3, 0.3, 0.08);
    const N = rr.nx * rr.nz, total = async () => { const d2 = await M.debug.read('dye'); let m = 0, mx = 0, cx = 0; for (let c = 0; c < N; c++) { m += d2[c]; cx += d2[c] * ((c % rr.nx) + 0.5) * dx; } return { m, cx: cx / m }; };
    const a = await total(); await M.debug.steps(Math.round(15 / stepFor(dx, 1.0)), stepFor(dx, 1.0)); const b = await total();
    check('dye is carried by the flow and its total is kept (within 1% over 15 s)', a.m > 0 && Math.abs(b.m / a.m - 1) < 0.01, 'total ' + (100 * (b.m / a.m - 1)).toFixed(3) + '%, centre moved ' + ((b.cx - a.cx) * 100).toFixed(1) + ' cm');
    const rerr2 = await M.debug.validateRender(400, 200);
    check('the water renderer (with dye, plumes and foam) builds and draws', !rerr2, rerr2);
    M.streak = true; M.exposure = 0.5; for (let j = 0; j < 5; j++) { M.stepParticles(0.02); }
    const rerr3 = await M.debug.validateRender(400, 200); M.streak = false;
    check('the long exposure (flakes, fading image, tone curve) builds and draws', !rerr3, rerr3);
  }

  // 24. smoke comb: in an empty river flume a filament from mid-depth stays level and its smoke
  // travels at about the mean velocity q / h
  if (want('comb')) {
    const zb = 0.02, h = 0.20, q = 0.06, rr = flume(6, 0.4, dx, (x, z) => (z < zb ? 1 : 0));
    await setup(rr, { type: 'river', h0: zb + h, q, hout: h });
    const dtp = stepFor(dx, 1.0); await M.debug.steps(Math.round(15 / dtp), dtp);
    M.comb.on = true; M.setComb(1.0);
    for (let j = 0; j < 240; j++) { await M.debug.steps(Math.max(1, Math.round((1 / 60) / dtp)), dtp); M.stepParticles(1 / 60); }
    const CB = await M.debug.read('CB'), C = M.debug.comb(), s = Math.floor(C.S / 2), zs = [], sp = [];
    for (let j = 0; j < C.M; j++) { const o = 4 * (s * C.M + j); if (CB[o + 2] > 0.5) { zs.push(CB[o + 1]); sp.push((CB[o] - 1.0) / CB[o + 2]); } }
    zs.sort((a, b) => a - b); sp.sort((a, b) => a - b);
    const spread = zs.length ? zs[zs.length - 1] - zs[0] : 1, v = sp.length ? sp[sp.length >> 1] : 0, U = q / h;
    check('smoke comb: a mid-depth filament stays level (within 1 cm) and travels at 0.8 to 1.3 times q/h', zs.length > 100 && spread < 0.01 && v / U > 0.8 && v / U < 1.3,
      zs.length + ' particles, height spread ' + (spread * 1000).toFixed(1) + ' mm, speed ' + (v * 100).toFixed(1) + ' cm/s vs q/h ' + (U * 100).toFixed(1) + ' cm/s');
    const rerr = await M.debug.validateRender(400, 200); M.comb.on = false;
    check('the smoke comb filaments build and draw', !rerr, rerr);
  }

  // 25. bubble wire: in an empty river flume a timeline's displacement over its age is the velocity
  // profile: each mark's speed matches the flow at its height, and the bed lags the surface
  if (want('wire')) {
    const zb = 0.02, h = 0.20, q = 0.06, rr = flume(6, 0.4, dx, (x, z) => (z < zb ? 1 : 0));
    await setup(rr, { type: 'river', h0: zb + h, q, hout: h });
    const dtp = stepFor(dx, 1.0); await M.debug.steps(Math.round(15 / dtp), dtp);
    M.wires.on = true; M.setWires([1.5]); const Tr = M.wireInterval();
    for (let t = 0; t < 2.5; t += 1 / 60) { await M.debug.steps(Math.max(1, Math.round((1 / 60) / dtp)), dtp); M.stepParticles(1 / 60); }
    const WB = await M.debug.read('WB'), Wc = M.debug.wire(), u = await M.debug.read('u');
    // the oldest live timeline
    let best = -1, bestAge = 0; for (let r = 0; r < Wc.R; r++) { const a = WB[4 * (r * Wc.K) + 2]; if (a > bestAge) { bestAge = a; best = r; } }
    let worst = 0, n = 0; const sp = [];
    for (let k = 0; k < Wc.K; k++) {
      const o = 4 * (best * Wc.K + k); if (WB[o + 2] <= 0) continue;
      const xs = WB[o], zs = WB[o + 1], v = (xs - 1.5) / WB[o + 2];
      const i = Math.round(((xs + 1.5) / 2) / dx), kk = Math.floor(zs / dx), uc = u[i + (rr.nx + 1) * kk];
      sp.push([zs, v]); if (zs > zb + 0.03 && zs < zb + h - 0.03) { worst = Math.max(worst, Math.abs(v / uc - 1)); n++; }
    }
    const lowV = sp.length ? sp[0][1] : 0, topV = sp.length ? sp[sp.length - 1][1] : 0;
    check('bubble wire: each mark of a timeline travels at the flow speed at its height (within 10%), the bed lagging the surface', n > 20 && worst < 0.10 && lowV < 0.85 * topV,
      'timeline age ' + bestAge.toFixed(2) + ' s (release every ' + Tr.toFixed(2) + ' s), ' + n + ' marks, worst ' + (worst * 100).toFixed(1) + '%, bed ' + (lowV * 100).toFixed(1) + ' vs top ' + (topV * 100).toFixed(1) + ' cm/s');
    const rerr = await M.debug.validateRender(400, 200); M.wires.on = false;
    check('the bubble wire timelines build and draw', !rerr, rerr);
  }

  // 23. restart: a river with a moving bed, dye and suspended sand, then reset with a bed readback in
  // flight; everything goes back to the design and the stale readback is not applied
  if (want('restart')) {
    const zb = 0.10, rr = flume(6, 0.5, dx, (x, z) => (z < zb ? 4 : (x > 2.5 && x < 2.8 && z < 0.2) ? 1 : 0));
    await setup(rr, { type: 'river', h0: 0.30, q: 0.10, hout: 0.22, sediment: true, morfac: 30 });
    const dtp = stepFor(dx, 1.0); M.injectDye(1.5, 0.2, 0.06); await M.debug.steps(Math.round(20 / dtp), dtp);
    const N = rr.nx * rr.nz, sum = (a) => { let m = 0; for (const v of a) m += Math.abs(v); return m; };
    const csBefore = sum(await M.debug.read('cs')), bedA = await M.debug.read('bed');
    let moved = 0; for (let i = 0; i < rr.nx; i++) moved = Math.max(moved, Math.abs(bedA[3 * i] - M.bed0.zs[i]));
    let applied = 0; const api0 = M.api; M.api = Object.assign({}, api0, { bedChanged: () => { applied++; } });
    M.debug.readBed(); M.reset();
    await M.device.queue.onSubmittedWorkDone(); await new Promise(r => setTimeout(r, 200));
    const staleApplied = applied;
    M.debug.readBed(); await M.device.queue.onSubmittedWorkDone(); await new Promise(r => setTimeout(r, 200));
    M.api = api0;
    const bedB = await M.debug.read('bed'); let back = 0; for (let i = 0; i < rr.nx; i++) back = Math.max(back, Math.abs(bedB[3 * i] - M.bed0.zs[i]));
    const left = { cs: sum(await M.debug.read('cs')), dye: sum(await M.debug.read('dye')), u: sum(await M.debug.read('u')), w: sum(await M.debug.read('w')), nt: sum(await M.debug.read('nt')) };
    check('restart clears the run: bed back to the design, no sand, dye or motion left in the water', moved > 0.002 && csBefore > 0 && back < 1e-6 && left.cs === 0 && left.dye === 0 && left.u === 0 && left.w === 0 && left.nt === 0 && M.t === 0,
      'bed had moved ' + (moved * 100).toFixed(1) + ' cm, back within ' + back.toExponential(1) + ' m; left: ' + JSON.stringify(left));
    check('a bed readback started before the restart is dropped (a new one is applied)', staleApplied === 0 && applied === 1, 'stale applied ' + staleApplied + ', new applied ' + (applied - staleApplied));
  }

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
})();
