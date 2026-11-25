// Wave-flume measurements on the GPU (dev tool, used by the model checks and for tuning).
//   node dev/wave_lab.js [T ...]      flat bed, h = 0.4 m, H = 0.04 m
// For each period: wavelength from the phase lag between gauges (against linear theory), the
// incident and reflected wave amplitudes in front of the far end (three-gauge least squares,
// Mansard & Funke), and the wave height next to the paddle.
'use strict';
const fs = require('fs'), path = require('path'), vm = require('vm');

// least-squares amplitude and phase of one frequency in a time series: eta ~ A cos wt + B sin wt
function harmonic(ts, ys, om) {
  let cc = 0, ss = 0, cs = 0, yc = 0, ysn = 0, m = 0; for (const y of ys) m += y / ys.length;
  for (let j = 0; j < ts.length; j++) { const c = Math.cos(om * ts[j]), s = Math.sin(om * ts[j]), y = ys[j] - m; cc += c * c; ss += s * s; cs += c * s; yc += y * c; ysn += y * s; }
  const det = cc * ss - cs * cs, A = (yc * ss - ysn * cs) / det, B = (ysn * cc - yc * cs) / det;
  return { re: A, im: -B };   // complex amplitude C with eta = Re[C e^{iwt}]; an incident wave has C = a e^{-ikx}
}
// incident/reflected split: Z_j = aI e^{-ik x_j} + aR e^{+ik x_j}, least squares over gauges
function separate(Z, xs, k) {
  // normal equations for complex unknowns aI, aR
  let s11 = 0, s12r = 0, s12i = 0, s22 = 0, b1r = 0, b1i = 0, b2r = 0, b2i = 0;
  for (let j = 0; j < xs.length; j++) {
    const e1r = Math.cos(-k * xs[j]), e1i = Math.sin(-k * xs[j]), e2r = Math.cos(k * xs[j]), e2i = Math.sin(k * xs[j]);
    s11 += 1; s22 += 1;
    // conj(e1) * e2
    s12r += e1r * e2r + e1i * e2i; s12i += e1r * e2i - e1i * e2r;
    b1r += e1r * Z[j].re + e1i * Z[j].im; b1i += e1r * Z[j].im - e1i * Z[j].re;
    b2r += e2r * Z[j].re + e2i * Z[j].im; b2i += e2r * Z[j].im - e2i * Z[j].re;
  }
  // [s11 s12; conj(s12) s22] [aI; aR] = [b1; b2]
  const detr = s11 * s22 - (s12r * s12r + s12i * s12i);
  const aIr = (s22 * b1r - (s12r * b2r - s12i * b2i)) / detr, aIi = (s22 * b1i - (s12r * b2i + s12i * b2r)) / detr;
  const aRr = (s11 * b2r - (s12r * b1r + s12i * b1i)) / detr, aRi = (s11 * b2i - (s12r * b1i - s12i * b1r)) / detr;
  return { aI: Math.hypot(aIr, aIi), aR: Math.hypot(aRr, aRi) };
}

async function measure(M, o) {
  const { L = 25, Hd = 0.8, dx = 0.02, h = 0.4, H = 0.04, T, paint } = o;
  const nx = Math.round(L / dx), nz = Math.round(Hd / dx), N = nx * nz;
  const r = { nx, nz, dx, mat: new Uint8Array(N), p1: new Float32Array(N), p2: new Float32Array(N), p3: new Float32Array(N), cls: new Uint8Array(N) };
  if (paint) for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) { const m = paint((i + .5) * dx, (k + .5) * dx); if (m) r.mat[i + nx * k] = m; }
  M.setScale('lab'); Object.assign(M.p, { type: 'wave', h0: h, H, T }); M.setGrid({ nx, nz, dx }, r); M.reset();
  const wl = M.debug.pistonStroke(H, T, h), k = 2 * Math.PI / wl.L, om = 2 * Math.PI / T;
  const sx0 = M.P.sx0;
  // gauges: a pair one quarter wavelength apart for the phase speed, three for the reflection split
  const gx = o.gauges || [3, 3 + wl.L / 4, sx0 - 0.9 * wl.L, sx0 - 0.9 * wl.L + 0.22 * wl.L, sx0 - 0.9 * wl.L + 0.5 * wl.L, 0.3];
  const dt = 0.4 * dx / 1.0, settle = o.settle || (sx0 / (wl.L / T) * 2 + 6 * T), rec = o.record || 6 * T;
  await M.debug.steps(Math.round(settle / dt), dt);
  const every = Math.max(1, Math.round(T / 32 / dt)), ts = [], ys = gx.map(() => []);
  for (let t = 0; t < rec; t += every * dt) {
    await M.debug.steps(every, dt); ts.push(M.t);
    const eta = await M.debug.read('eta'); gx.forEach((x, j) => ys[j].push(eta[Math.round(x / dx)] - (o.still || h)));
  }
  const Z = ys.map(y => harmonic(ts, y, om));
  // phase lag between the first two gauges -> wavelength
  const ph = (z) => Math.atan2(z.im, z.re);
  let dphi = ph(Z[0]) - ph(Z[1]); while (dphi < 0) dphi += 2 * Math.PI; while (dphi > 2 * Math.PI) dphi -= 2 * Math.PI;
  const Lmeas = 2 * Math.PI * (gx[1] - gx[0]) / dphi;
  const sep = separate(Z.slice(2, 5), gx.slice(2, 5), k);
  const Hpad = 2 * Math.hypot(Z[5].re, Z[5].im);
  return { T, Ltheory: wl.L, Lmeas, Hin: 2 * sep.aI, Kr: sep.aR / sep.aI, Hpad, sx0, Z, gx, ts, ys };
}
// incident height at three nearby gauges from the series already recorded (re-run the record)
async function measureNear(M, xs, m0, wl) {
  const om = 2 * Math.PI / m0.T, k = 2 * Math.PI / wl.L, dx = M.grid.dx, dt = 0.4 * dx / 1.0;
  const every = Math.max(1, Math.round(m0.T / 32 / dt)), ts = [], ys = xs.map(() => []);
  for (let t = 0; t < 6 * m0.T; t += every * dt) {
    await M.debug.steps(every, dt); ts.push(M.t);
    const eta = await M.debug.read('eta'); xs.forEach((x, j) => ys[j].push(eta[Math.round(x / dx)] - 0.4));
  }
  return 2 * separate(ys.map(y => harmonic(ts, y, om)), xs, k).aI;
}
// wavelength from the slope of phase against distance over many gauges (least squares, unwrapped)
function wavelengthFit(m) {
  const ph = m.Z.map(z => -Math.atan2(z.im, z.re));       // incident wave: phase = k x
  const un = [ph[0]]; for (let j = 1; j < ph.length; j++) { let d = ph[j] - ph[j - 1]; while (d < -Math.PI) d += 2 * Math.PI; while (d > Math.PI) d -= 2 * Math.PI; un.push(un[j - 1] + d); }
  const n = m.gx.length, mx = m.gx.reduce((a, b) => a + b, 0) / n, my = un.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0; for (let j = 0; j < n; j++) { sxy += (m.gx[j] - mx) * (un[j] - my); sxx += (m.gx[j] - mx) ** 2; }
  return 2 * Math.PI / (sxy / sxx);
}
module.exports = { measure, harmonic, separate, measureNear, wavelengthFit };

if (require.main === module) (async () => {
  const { create, globals } = await import('webgpu'); Object.assign(globalThis, globals);
  const opts = ['backend=vulkan']; if (process.env.FLUME_ADAPTER) opts.unshift('adapter=' + process.env.FLUME_ADAPTER);
  let M = null; globalThis.registerModel = (m) => { M = m; };
  vm.runInThisContext(fs.readFileSync(path.join(__dirname, '..', 'model.js'), 'utf8'), { filename: 'model.js' });
  await M.init({ gpu: create(opts) });
  if (process.argv[2] === 'absorb') {
    // a wall at the far end: the incident wave over time with and without paddle absorption
    const T = 1.6, H = 0.04, h = 0.4, dx = 0.02, L = 20, nx = L / dx, nz = 0.8 / dx, N = nx * nz;
    const wl = M.debug.pistonStroke(H, T, h), k = 2 * Math.PI / wl.L, om = 2 * Math.PI / T;
    const xs = [6, 6 + 0.22 * wl.L, 6 + 0.5 * wl.L], dt = 0.4 * dx / 1.0;
    for (const absorb of [false, true]) {
      const r = { nx, nz, dx, mat: new Uint8Array(N), p1: new Float32Array(N), p2: new Float32Array(N), p3: new Float32Array(N), cls: new Uint8Array(N) };
      M.setScale('lab'); Object.assign(M.p, { type: 'wave', h0: h, H, T, farEnd: 'wall', absorb }); M.setGrid({ nx, nz, dx }, r); M.reset();
      const out = [];
      for (const until of [20, 40, 60]) {
        await M.debug.steps(Math.round((until - 6 * T - M.t) / dt), dt);
        const every = Math.max(1, Math.round(T / 32 / dt)), ts = [], ys = xs.map(() => []);
        for (let t = 0; t < 6 * T; t += every * dt) { await M.debug.steps(every, dt); ts.push(M.t); const e = await M.debug.read('eta'); xs.forEach((x, j) => ys[j].push(e[Math.round(x / dx)] - h)); }
        const sp = separate(ys.map(y => harmonic(ts, y, om)), xs, k);
        out.push(`t ${until} s: incident ${(200 * sp.aI).toFixed(2)} cm, reflected ${(200 * sp.aR).toFixed(2)} cm`);
      }
      console.log((absorb ? 'paddle absorbs:   ' : 'no absorption:    ') + out.join(' | '));
    }
    process.exit(0);
  }
  if (process.argv[2] === 'wavelength') {
    for (const T of process.argv.slice(3).map(Number)) {
      const gauges = []; for (let x = 3; x <= 12.01; x += 0.25) gauges.push(x);
      const m = await measure(M, { T, H: +(process.env.WAVE_H || 0.04), gauges: gauges.concat([0, 0, 0, 0]).slice(0, gauges.length) });
      console.log(`T ${T} s, H ${process.env.WAVE_H || 0.04} m: wavelength fit ${wavelengthFit(m).toFixed(3)} m, linear theory ${m.Ltheory.toFixed(3)} m (${(100 * (wavelengthFit(m) / m.Ltheory - 1)).toFixed(1)}%)`);
    }
    process.exit(0);
  }
  if (process.argv[2] === 'decay') {
    // incident height near the paddle and near the far end, for variants of the model
    const T = +(process.argv[3] || 1.6), dx = +(process.argv[4] || 0.02);
    for (const v of (process.argv[5] || 'base,nofriction').split(',')) {
      M.wallFriction = v !== 'nofriction'; M.turbulence = v !== 'noturb'; M.maccormack = v !== 'nomac';
      const wl = M.debug.pistonStroke(0.04, T, 0.4);
      const near = [3, 3 + 0.22 * wl.L, 3 + 0.5 * wl.L];
      const m = await measure(M, { T, dx, gauges: [3, 3 + wl.L / 4, 14, 14 + 0.22 * wl.L, 14 + 0.5 * wl.L, 0.3] });
      const Zn = await measureNear(M, near, m, wl);
      console.log(`T ${T} dx ${dx} ${v}: H near (3 m) ${(Zn * 100).toFixed(2)} cm, H far (14 m) ${(m.Hin * 100).toFixed(2)} cm, loss ${(100 * (1 - m.Hin / Zn)).toFixed(1)}% over 11 m`);
    }
    process.exit(0);
  }
  const Ts = process.argv.slice(2).map(Number); if (!Ts.length) Ts.push(1.2, 1.6, 2.4);
  for (const T of Ts) {
    const m = await measure(M, { T });
    console.log(`T ${T} s: wavelength ${m.Lmeas.toFixed(3)} m (theory ${m.Ltheory.toFixed(3)}, ${(100 * (m.Lmeas / m.Ltheory - 1)).toFixed(1)}%), incident H ${(m.Hin * 100).toFixed(2)} cm (asked 4.00), reflection Kr ${m.Kr.toFixed(3)}, sponge from ${m.sx0.toFixed(1)} m`);
  }
  process.exit(0);
})();
