// Render a scene on the GPU (Dawn) to PNG frames and an MP4, for reviewing the look on real hardware.
//   node dev/render_movie.js <scene: jump|waves> <out dir> [seconds] [fps]
//   env: WARM (s of spin-up, default 4), STREAK=<exposure s> (long exposure on), COMB=1|<x m> (smoke comb on, at x), LIGHT, GAIN (its tuning), PARTS=tracers|foam|none (draw one kind only, for tuning), FRAMES=a,b (stills only)
// Water and particles come from model.js's renderer (offscreen); solids and sediment are drawn on the
// CPU the way the page draws them (black background, sand, grey solids), then ffmpeg makes the MP4.
'use strict';
const fs = require('fs'), path = require('path'), vm = require('vm'), zlib = require('zlib'), cp = require('child_process');

function png(file, W, H, rgba) {
  const raw = Buffer.alloc((W * 4 + 1) * H);
  for (let y = 0; y < H; y++) { raw[y * (W * 4 + 1)] = 0; rgba.copy ? rgba.copy(raw, y * (W * 4 + 1) + 1, y * W * 4, (y + 1) * W * 4) : raw.set(rgba.subarray(y * W * 4, (y + 1) * W * 4), y * (W * 4 + 1) + 1); }
  const crcT = []; for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; crcT[n] = c >>> 0; }
  const crc = (b) => { let c = 0xFFFFFFFF; for (const x of b) c = crcT[(c ^ x) & 255] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; };
  const chunk = (t, d) => { const len = Buffer.alloc(4); len.writeUInt32BE(d.length); const td = Buffer.concat([Buffer.from(t), d]); const c = Buffer.alloc(4); c.writeUInt32BE(crc(td)); return Buffer.concat([len, td, c]); };
  const ih = Buffer.alloc(13); ih.writeUInt32BE(W, 0); ih.writeUInt32BE(H, 4); ih[8] = 8; ih[9] = 6; ih[10] = 0; ih[11] = 0; ih[12] = 0;
  fs.writeFileSync(file, Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ih), chunk('IDAT', zlib.deflateSync(raw, { level: 3 })), chunk('IEND', Buffer.alloc(0))]));
}

const SCENES = {
  // a sluice gate and a hydraulic jump, then a sill downstream
  jump: { L: 8, H: 0.6, dx: 0.01, p: { type: 'river', h0: 0.07, q: 0.10, hout: 0.19, sediment: false },
    paint: (x, z) => (z < 0.02 ? 1 : (x < 0.15 && z > 0.06) ? 1 : (x > 5.6 && x < 5.9 && z < 0.10) ? 1 : 0), view: { x0: -0.12, z0: -0.025, L: 4.8, ve: 3, H: 440 }, crests: [[5.6, 5.9, 0.10]] },
  // subcritical flow over a block: lift over the crest and the separation eddy behind it
  block: { L: 6, H: 0.5, dx: 0.008, p: { type: 'river', h0: 0.30, q: 0.07, hout: 0.27, sediment: false },
    paint: (x, z) => (z < 0.02 ? 1 : (x > 2.4 && x < 2.7 && z < 0.16) ? 1 : 0), view: { x0: 0.6, z0: -0.01, L: 4.2, ve: 2, H: 420 }, crests: [[2.4, 2.7, 0.16]] },
  // waves over a submerged reef breaking on a sand beach
  waves: { L: 20, H: 1.0, dx: 0.0125, p: { type: 'wave', h0: 0.45, H: 0.12, T: 2.2, farEnd: 'beach', absorb: true, sediment: false },
    paint: (x, z) => { const bed = x < 9 ? 0.06 : 0.06 + (x - 9) * 0.07; if (z < bed) return 4; if (x > 6.5 && x < 7.8 && z < 0.32 - Math.abs(x - 7.15) * 0.3) return 1; return 0; }, view: { x0: 5.2, z0: -0.04, L: 10.8, ve: 3, H: 400 }, crests: [[6.5, 7.8, 0.32]] },
};

(async () => {
  const which = process.argv[2] || 'jump', out = process.argv[3], secs = +(process.argv[4] || 12), fps = +(process.argv[5] || 30);
  const S = SCENES[which]; fs.mkdirSync(out, { recursive: true });
  if (process.env.VIEWJSON) fs.writeFileSync(process.env.VIEWJSON, JSON.stringify(Object.assign({ crests: S.crests || [] }, S.view)));
  const { create, globals } = await import('webgpu'); Object.assign(globalThis, globals);
  let M = null; globalThis.registerModel = (m) => { M = m; };
  vm.runInThisContext(fs.readFileSync(path.join(__dirname, '..', 'model.js'), 'utf8'), { filename: 'model.js' });
  const res = await M.init({ gpu: create(['adapter=' + (process.env.FLUME_ADAPTER || '4070'), 'backend=vulkan']) }); if (!res.ok) { console.log(res.reason); process.exit(1); }
  const nx = Math.round(S.L / S.dx), nz = Math.round(S.H / S.dx), N = nx * nz, dx = S.dx;
  const r = { nx, nz, dx, mat: new Uint8Array(N), p1: new Float32Array(N), p2: new Float32Array(N), p3: new Float32Array(N), p4: new Float32Array(N), cls: new Uint8Array(N), frac: new Float32Array(N) };
  // material by the cell centre, coverage by 4x4 samples (smooth outlines, as the page's painter)
  for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) {
    const m = S.paint((i + 0.5) * dx, (k + 0.5) * dx); let n = 0;
    for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) if (S.paint((i + (a + 0.5) / 4) * dx, (k + (b + 0.5) / 4) * dx)) n++;
    if (m) { r.mat[i + nx * k] = m; r.frac[i + nx * k] = n / 16; }
  }
  M.setScale('lab'); Object.assign(M.p, S.p); M.setGrid({ nx, nz, dx }, r); M.reset();
  const W = 1600, H = +(process.env.H || S.view.H || 480), v = S.view, sc = W / v.L, view = { x0: v.x0, z0: v.z0, s: sc, ve: v.ve, cw: W, ch: H, dpr: 1 };
  // solids and sand, drawn once (black where not solid), as the page would
  // the section cut, as the page draws it: nothing inside solids and sediment (black), a pink line on their edge
  const inS = new Uint8Array(W * H), mat = new Uint8ClampedArray(W * H * 4);
  for (let py = 0; py < H; py++) for (let px = 0; px < W; px++) inS[px + W * py] = S.paint(v.x0 + (px + 0.5) / sc, v.z0 + (H - py - 0.5) / (sc * v.ve)) ? 1 : 0;
  for (let py = 0; py < H; py++) for (let px = 0; px < W; px++) {
    const o = 4 * (px + W * py), me = inS[px + W * py];
    let edge = false;
    for (let dy = -1; dy <= 1 && !edge; dy++) for (let dx2 = -1; dx2 <= 1; dx2++) { const qx = px + dx2, qy = py + dy; if (qx >= 0 && qy >= 0 && qx < W && qy < H && inS[qx + W * qy] !== me) { edge = true; break; } }
    if (edge) { mat[o] = 255; mat[o + 1] = 46; mat[o + 2] = 147; mat[o + 3] = 255; }
    else if (me) { mat[o + 3] = 255; }
  }
  // offscreen target and readback
  const fmt = 'rgba8unorm', tex = M.device.createTexture({ size: [W, H], format: fmt, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const bpr = Math.ceil(W * 4 / 256) * 256, rb = M.device.createBuffer({ size: bpr * H, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  M.format = fmt; M.render_ = null;
  const parts = process.env.PARTS; if (parts === 'none') M.particles = false;
  if (process.env.COMB) { M.comb.on = true; if (process.env.COMB !== '1') M.setComb(+process.env.COMB); }
  if (process.env.STREAK) { M.streak = true; M.exposure = +process.env.STREAK; if (process.env.LIGHT) M.expLight = +process.env.LIGHT; if (process.env.GAIN) M.expGain = +process.env.GAIN; }
  else if (parts) M.drawKinds = parts;
  const dt = 0.4 * dx / 1.5, perFrame = 1 / fps, warm = +(process.env.WARM || 4);
  let frame = 0;
  for (let t = -warm; t < secs; t += perFrame) {
    const n = Math.max(1, Math.round(perFrame / dt));
    await M.debug.steps(n, dt);
    M.stepParticles(n * dt);                     // the particles move by this frame's model time
    if (t < 0) { if (process.env.SECTION && frame % 1 === 0) await M.debug.readColumns(); continue; }
    const skip = process.env.FRAMES && !process.env.FRAMES.split(',').includes(String(frame));
    if (skip && !M.streak) { frame++; continue; }
    if (process.env.SECTION) { M.wantSection = true; await M.debug.readColumns(); fs.writeFileSync(path.join(out, 's' + String(frame).padStart(5, '0') + '.json'), JSON.stringify(M.sectionLines())); }
    M.render(Object.assign({}, view, { target: tex, _rb: true }));
    if (skip) { await M.device.queue.onSubmittedWorkDone(); frame++; continue; }
    const e = M.device.createCommandEncoder(); e.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow: bpr }, [W, H]); M.device.queue.submit([e.finish()]);
    await rb.mapAsync(GPUMapMode.READ); const src = new Uint8Array(rb.getMappedRange());
    const img = Buffer.alloc(W * H * 4);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const so = y * bpr + x * 4, o = 4 * (x + W * y), a = src[so + 3] / 255;
      // black, then water and particles (premultiplied), then solids on top
      let R = src[so], G = src[so + 1], B = src[so + 2];
      const ma = mat[o + 3] / 255; R = R * (1 - ma) + mat[o] * ma; G = G * (1 - ma) + mat[o + 1] * ma; B = B * (1 - ma) + mat[o + 2] * ma;
      img[o] = R; img[o + 1] = G; img[o + 2] = B; img[o + 3] = 255;
    }
    rb.unmap();
    png(path.join(out, 'f' + String(frame++).padStart(5, '0') + '.png'), W, H, img);
  }
  if (!process.env.FRAMES && !process.env.NOVIDEO) cp.execFileSync('ffmpeg', ['-loglevel', 'error', '-y', '-framerate', String(fps), '-i', path.join(out, 'f%05d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', path.join(out, which + '.mp4')]);
  console.log('wrote', frame, 'frames and', path.join(out, which + '.mp4'));
  process.exit(0);
})();
