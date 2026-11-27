// Headless checks for the flume interface (index.html), under a stub model.
// Takes the inline interface script, runs it with a small DOM mock, registers a stub model
// that records what the interface asks of it, boots, and drives the app like a person would.
//   node dev/interface_checks.js
'use strict';
const fs = require('fs'), path = require('path'), vm = require('vm');
const ROOT = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const src = html.match(/<script id="interface">([\s\S]*?)<\/script>/)[1];
const markupIds = new Set([...html.replace(src, '').matchAll(/\sid="([^"]+)"/g)].map(m => m[1]));

// ───────── DOM mock ─────────
const W = 1200, H = 600;
const ctxProxy = () => new Proxy({ createImageData: (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
  createLinearGradient: () => ({ addColorStop() {} }), measureText: () => ({ width: 10 }) },
  { get: (t, k) => (k in t ? t[k] : () => {}), set: (t, k, v) => { t[k] = v; return true; } });
const missingIds = new Set();
function makeEl(id, tag = 'div') {
  const el = {
    id, tagName: tag.toUpperCase(), style: {}, dataset: {}, hidden: false, value: '', checked: false, textContent: '', files: null,
    _html: '', listeners: {}, children: [], width: 0, height: 0,
    classList: { _s: new Set(), add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); }, toggle(c, on) { (on === undefined ? !this._s.has(c) : on) ? this._s.add(c) : this._s.delete(c); }, contains(c) { return this._s.has(c); } },
    addEventListener(t, fn) { (this.listeners[t] = this.listeners[t] || []).push(fn); },
    dispatch(t, e = {}) { for (const fn of this.listeners[t] || []) fn(Object.assign({ target: this, preventDefault() {} }, e)); },
    getContext: () => el._ctx || (el._ctx = ctxProxy()),
    getBoundingClientRect: () => ({ left: 0, top: 0, width: W, height: H }),
    setPointerCapture() {}, appendChild() {}, remove() {}, click() { this.dispatch('click'); if (this.download) downloads.push(this.download); },
    toBlob(cb) { cb({ size: 1 }); },
    querySelectorAll(sel) { return queryAll(sel, this); },
  };
  Object.defineProperty(el, 'innerHTML', { get() { return el._html; }, set(v) { el._html = v; parseInto(v); } });
  return el;
}
const els = {}, downloads = [];
function getEl(id) {
  if (!els[id]) {
    if (!markupIds.has(id) && !dynamicIds.has(id)) missingIds.add(id);
    els[id] = makeEl(id);
    const tag = html.match(new RegExp('<(\\w+)[^>]*\\sid="' + id + '"[^>]*>'));
    if (tag) { els[id].tagName = tag[1].toUpperCase(); const val = tag[0].match(/\svalue="([^"]*)"/); if (val) els[id].value = val[1]; }
  }
  return els[id];
}
// ids created by innerHTML (model controls, material properties) become real elements too
const dynamicIds = new Set();
const dataKeyed = [];
function parseInto(htmlText) {
  for (const m of htmlText.matchAll(/<(\w+)[^>]*\sid="([^"]+)"[^>]*>/g)) {
    dynamicIds.add(m[2]); const el = getEl(m[2]); el.tagName = m[1].toUpperCase();
    const dk = m[0].match(/data-k="([^"]+)"/); if (dk) { el.dataset.k = dk[1]; dataKeyed.push(el); }
    const cls = m[0].match(/class="([^"]+)"/); if (cls) for (const c of cls[1].split(/\s+/)) el.classList.add(c);
    const val = m[0].match(/value="([^"]*)"/); if (val && m[1] !== 'option') el.value = val[1];
  }
}
const toolButtons = [...html.matchAll(/data-tool="(\w+)"/g)].map(m => { const b = makeEl('', 'button'); b.dataset.tool = m[1]; return b; });
function queryAll(sel, within) {
  if (sel === '#toolbar button') return toolButtons;
  if (sel === '[data-k]') return dataKeyed.filter(e => within && within._html.includes('id="' + e.id + '"'));
  if (sel.startsWith('.')) return Object.values(els).filter(e => e.classList.contains(sel.slice(1)));
  return [];
}
const winListeners = {};
let rafCb = null;
const store = new Map();
const sandbox = {
  console, Math, JSON, Date, Object, Array, Set, Map, Uint8Array, Uint32Array, Float32Array, Float64Array, Uint8ClampedArray, Promise, Error, parseFloat, parseInt, isFinite, String, Number, Proxy, setTimeout, clearTimeout,
  performance: { now: () => Date.now() },
  document: { getElementById: getEl, createElement: (t) => makeEl('', t), querySelectorAll: (s) => queryAll(s), body: { appendChild() {} } },
  window: { addEventListener: (t, fn) => { (winListeners[t] = winListeners[t] || []).push(fn); }, devicePixelRatio: 1 },
  requestAnimationFrame: (cb) => { rafCb = cb; },
  localStorage: { getItem: (k) => (store.has(k) ? store.get(k) : null), setItem: (k, v) => store.set(k, String(v)), removeItem: (k) => store.delete(k) },
  URL: { createObjectURL: () => 'blob:', revokeObjectURL() {} }, Blob: function () {}, alert: () => {},
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

// ───────── stub model ─────────
const STUB = `
const calls = { watch: [], dye: [], init: 0, setGrid: 0, setRaster: 0, reset: 0, frame: 0, framesRunning: 0, render: 0, setScale: [] };
registerModel({
  id: 'stub', calls, p: { speed: 1 },
  async init(api) { calls.init++; calls.api = api; return { ok: false, reason: 'stub model' }; },
  domain(scale) { return scale === 'field' ? { L: 400, H: 16, dx: 0.2 } : { L: 20, H: 1.0, dx: 0.01 }; },
  setScale(s) { calls.setScale.push(s); },
  setGrid(g, r) { calls.setGrid++; calls.grid = g; calls.raster = r; },
  setRaster(r) { calls.setRaster++; calls.raster = r; },
  reset() { calls.reset++; },
  frame(dt, running) { calls.frame++; if (running) calls.framesRunning++; },
  render(v) { calls.render++; calls.view = v; },
  readouts() { return [['stub', 'ok']]; },
  watchSurface(on) { calls.watch.push(on); },
  surfaceSample() { return calls.sample || null; },
  gaugeWindow() { return 10; },
  injectDye(x, z, r) { calls.dye.push([x, z, r]); }, clearDye() { calls.dyeCleared = (calls.dyeCleared || 0) + 1; },
  state() { return { stubState: 7 }; }, setState(s) { calls.state = s; },
  controls: { html: '<div class="row"><label for="stub-x">x</label><input id="stub-x" type="number"></div>' },
});`;

// ───────── checks ─────────
let pass = 0, fail = 0;
function check(name, ok, detail) { if (ok) { pass++; console.log('  ok  ' + name); } else { fail++; console.log('  FAIL ' + name + (detail ? '  — ' + detail : '')); } }
const run = (code) => vm.runInContext(code, sandbox);

(async () => {
  console.log('flume interface checks');
  run(src + '\n;globalThis.__T = { get DOMAIN() { return DOMAIN; }, get GRID() { return GRID; }, get RASTER() { return RASTER; }, get scene() { return scene; }, get view() { return view; }, get model() { return model; }, get tool() { return tool; }, get selected() { return selected; }, get running() { return running; }, get undoStack() { return undoStack; }, get backends() { return backends; }, MAT, rasterize, rasterHash, w2s, s2w, zoomAt, setVE, fitView, contains, bboxOf, setTool, setDomain, setScale, captureScene, sceneFileText, openSceneText, applyScene, validScene, bedZ, onDown, onMove, onUp, onKey, undo, addShape, deleteSelected, reorderSelected, bedSlope, bedFlat, bedSmooth, setRunning, frame, saveLocal, loadLocal, STORE_KEY, updatePanel, get curMat() { return curMat; } };');
  run(STUB);
  run("registerBackend({ id: 'test-site', title: 'Test Site', mounted: 0, frames: 0, mount() { this.mounted++; }, frame() { this.frames++; } });");
  const T = sandbox.__T, M = T.model, C = M.calls;

  // 1. boot
  let bootErr = null;
  try { await run('boot()'); } catch (e) { bootErr = e; }
  check('boot completes', !bootErr && run('booted') === true, bootErr && bootErr.stack);
  check('the model is initialised once, given the GPU canvas, a change callback and a way to show a note', C.init === 1 && C.api && C.api.canvas === getEl('gpu') && typeof C.api.changed === 'function' && typeof C.api.notify === 'function');
  check('a model that cannot run says so in the banner', getEl('banner').hidden === false && getEl('banner').textContent === 'stub model');
  check('the model gets a grid and a raster at boot', C.setGrid >= 1 && C.raster && C.raster.nx === T.GRID.nx && C.raster.mat.length === T.GRID.nx * T.GRID.nz);
  check('model controls are mounted', getEl('model-controls').innerHTML.includes('stub-x'));
  check('the site backend is mounted once and sets the title', T.backends[0].mounted === 1);
  check('every id the script asks for exists in the page or in mounted markup', missingIds.size === 0, [...missingIds].join(', '));

  // 2. grid and default bed
  const g = T.GRID;
  check('normal resolution on the lab flume: 1.4 cm cells', Math.abs(g.dx - 0.014) < 0.001 && g.nx === Math.round(20 / g.dx), JSON.stringify(g));
  const R = T.RASTER, bedK = Math.round(0.1 / g.dx);
  let bedOk = true; for (let i = 0; i < g.nx; i += 37) { if (R.mat[i + g.nx * (bedK - 1)] !== 4 || R.mat[i + g.nx * (bedK + 1)] !== 0) bedOk = false; }
  check('the default bed is sediment up to 10% of the height, water above', bedOk);

  // 3. rasterizing shapes
  const before = C.setRaster;
  const rect = T.addShape('rect', [[5, 0.3], [6, 0.5]]);
  const cell = (x, z) => R.mat[Math.floor(x / g.dx) + g.nx * Math.floor(z / g.dx)];
  check('a rectangle fills its cells with the current material (solid)', cell(5.5, 0.4) === 1 && cell(4.9, 0.4) === 0 && cell(5.5, 0.55) === 0);
  check('the model is told when the materials change', C.setRaster === before + 1 && C.raster === T.RASTER);
  run('curMat = 2'); const ell = T.addShape('ellipse', [[8, 0.2], [10, 0.6]]);
  check('an ellipse fills an ellipse (porous, with its properties)', cell(9, 0.4) === 2 && cell(8.05, 0.21) === 0 && Math.abs(R.p1[Math.floor(9 / g.dx) + g.nx * Math.floor(0.4 / g.dx)] - 0.4) < 1e-6);
  run('curMat = 3'); T.addShape('poly', [[12, 0.1], [14, 0.1], [13, 0.5]]);
  check('a polygon fills a triangle (canopy)', cell(13, 0.2) === 3 && cell(12.1, 0.45) === 0);
  run('curMat = 3'); T.addShape('rect', [[3, 0.02], [4, 0.3]]);
  check('plants drawn into the bed take only the water above it', cell(3.5, 0.05) === 4 && cell(3.5, 0.2) === 3);
  run('curMat = 2'); T.addShape('rect', [[5.2, 0.35], [5.8, 0.6]]);
  check('porous rock drawn over a solid leaves the solid in place', cell(5.5, 0.4) === 1 && cell(5.5, 0.55) === 2);
  T.undo(); T.undo();
  run('curMat = 1'); T.addShape('stroke', [[15, 0.3], [17, 0.3]], 0.1);
  check('a stroke fills a band of its width', cell(16, 0.3) === 1 && cell(16, 0.34) === 1 && cell(16, 0.37) === 0);
  run("tool = 'erase'"); T.addShape('rect', [[5.4, 0.0], [5.6, 0.6]]); run("tool = 'select'");
  check('the eraser carves water, through shapes and the bed', cell(5.5, 0.4) === 0 && cell(5.5, 0.05) === 0 && cell(5.3, 0.4) === 1);
  T.undo();
  check('undo brings the carved cells back', cell(5.5, 0.4) === 1 && cell(5.5, 0.05) === 4);

  // 4. selection, move, resize, order, delete — through the pointer, as a person would
  T.setTool('select');
  const [Xc, Yc] = T.w2s(5.5, 0.4);
  T.onDown({ offsetX: Xc, offsetY: Yc, button: 0 });
  check('a click inside a shape selects it', T.selected === rect.id);
  const [Xm, Ym] = T.w2s(6.5, 0.6); T.onMove({ offsetX: Xm, offsetY: Ym }); T.onUp({});
  check('dragging moves the shape and its cells', cell(6.5, 0.6) === 1 && cell(5.2, 0.32) === 0, 'cells ' + cell(6.5, 0.6) + ' ' + cell(5.2, 0.32));
  const shp = T.scene.shapes.find(s => s.id === rect.id), bb = T.bboxOf(shp);
  const [Xh, Yh] = T.w2s(bb.x1, bb.z0);      // bottom-right handle
  T.onDown({ offsetX: Xh, offsetY: Yh, button: 0 }); const [Xr, Yr] = T.w2s(bb.x1 + 1, bb.z0); T.onMove({ offsetX: Xr, offsetY: Yr }); T.onUp({});
  const bb2 = T.bboxOf(shp);
  check('dragging a corner handle resizes about the opposite corner', Math.abs(bb2.x1 - (bb.x1 + 1)) < 1e-6 && Math.abs(bb2.x0 - bb.x0) < 1e-6 && Math.abs(bb2.z1 - bb.z1) < 1e-6);
  const idx = () => T.scene.shapes.findIndex(s => s.id === rect.id);
  const i0 = idx(); T.onKey({ key: ']', target: {}, preventDefault() {} });
  check('] moves the selected shape forward in the drawing order', idx() === i0 + 1);
  T.onKey({ key: 'Delete', target: {}, preventDefault() {} });
  check('Delete removes the selected shape', !T.scene.shapes.some(s => s.id === rect.id) && T.selected === null);
  check('a click in empty water selects nothing', (T.onDown({ offsetX: T.w2s(1, 0.8)[0], offsetY: T.w2s(1, 0.8)[1], button: 0 }), T.selected === null));

  // 5. drawing tools through the pointer
  T.setTool('rect'); const n0 = T.scene.shapes.length;
  let a = T.w2s(2, 0.3), b = T.w2s(3, 0.5);
  T.onDown({ offsetX: a[0], offsetY: a[1], button: 0 }); T.onMove({ offsetX: b[0], offsetY: b[1] }); T.onUp({});
  check('the rectangle tool draws a rectangle by dragging', T.scene.shapes.length === n0 + 1 && T.scene.shapes[n0].kind === 'rect' && cell(2.5, 0.4) === 1);
  T.onDown({ offsetX: a[0], offsetY: a[1], button: 0 }); T.onUp({});
  check('a click without a drag adds no rectangle', T.scene.shapes.length === n0 + 1);
  T.setTool('poly');
  for (const [x, z] of [[18, 0.5], [19, 0.5], [18.5, 0.8]]) { const p = T.w2s(x, z); T.onDown({ offsetX: p[0], offsetY: p[1], button: 0 }); }
  const pf = T.w2s(18, 0.5); T.onDown({ offsetX: pf[0], offsetY: pf[1], button: 0 });
  check('the polygon tool closes on a click at the first corner', T.scene.shapes.length === n0 + 2 && T.scene.shapes[n0 + 1].kind === 'poly' && cell(18.5, 0.6) === 1);
  T.setTool('brush');
  a = T.w2s(1, 0.5); b = T.w2s(3, 0.5);
  T.onDown({ offsetX: a[0], offsetY: a[1], button: 0 });
  for (let j = 1; j <= 20; j++) T.onMove({ offsetX: a[0] + (b[0] - a[0]) * j / 20, offsetY: a[1] });
  T.onUp({});
  const st = T.scene.shapes[T.scene.shapes.length - 1];
  check('the brush records a stroke with several points and the brush width', st.kind === 'stroke' && st.pts.length > 5 && Math.abs(st.w - 14 / T.view.s) < 1e-4 && cell(2, 0.5) === 1);

  // 6. bed
  T.setTool('bed');
  const nb = T.scene.bed.pts.length, [Xb, Yb] = T.w2s(10, T.bedZ(10));
  T.onDown({ offsetX: Xb, offsetY: Yb, button: 0 }); const [Xb2, Yb2] = T.w2s(10, 0.3); T.onMove({ offsetX: Xb2, offsetY: Yb2 }); T.onUp({});
  check('a click on the bed line adds a point and dragging it reshapes the bed', T.scene.bed.pts.length === nb + 1 && Math.abs(T.bedZ(10) - 0.3) < 0.002 && cell(10, 0.25) === 4);
  T.onDown({ offsetX: Xb2, offsetY: Yb2, button: 2 });
  check('a right-click removes a bed point', T.scene.bed.pts.length === nb);
  T.bedSlope(1);
  check('a 1% slope drops the bed by 1% of the length about its mean height', Math.abs(T.bedZ(0) - T.bedZ(20) - 0.2) < 1e-9 && Math.abs((T.bedZ(0) + T.bedZ(20)) / 2 - 0.1) < 1e-9);
  T.bedFlat();
  check('flat makes the bed level again', T.scene.bed.pts.length === 2 && T.bedZ(0) === T.bedZ(20));

  // 7. view
  T.fitView(); const v = T.view;
  const [Xp, Yp] = T.w2s(7.3, 0.42), [xb, zb] = T.s2w(Xp, Yp);
  check('screen and world coordinates round-trip', Math.abs(xb - 7.3) < 1e-9 && Math.abs(zb - 0.42) < 1e-9);
  check('the fitted view shows the whole flume with gentle vertical exaggeration (1 to 3)', T.w2s(0, 0)[0] >= 0 && T.w2s(20, 0)[0] <= v.cw && v.ve > 1 && v.ve <= 3);
  const [wx, wz] = T.s2w(400, 300); T.zoomAt(400, 300, 3); const [wx2, wz2] = T.s2w(400, 300);
  check('wheel zoom keeps the point under the pointer in place', Math.abs(wx - wx2) < 1e-9 && Math.abs(wz - wz2) < 1e-9);
  const [, vz] = T.s2w(400, 300); T.setVE(400, 300, 2); const [, vz2] = T.s2w(400, 300);
  check('changing the vertical scale keeps the height under the pointer', Math.abs(vz - vz2) < 1e-9);
  T.onKey({ key: '0', target: {}, preventDefault() {} });
  check('0 fits the view again', Math.abs(T.view.s - v.cw / (20 * 1.04)) < 1e-9);

  // 8. save, open, damaged files
  T.setTool('select');
  const hash = T.rasterHash(), text = T.sceneFileText();
  check('the scene file names the app and carries the model state', /"app": "flume-interface"/.test(text) && /"stubState": 7/.test(text));
  run('scene.shapes = []; rasterize();');
  check('opening the file brings the scene back cell for cell', T.openSceneText(text) && T.rasterHash() === hash && C.state && C.state.stubState === 7);
  check('a file that is not a scene is refused and changes nothing', !T.openSceneText('{"app":"other"}') && !T.openSceneText('not json') && T.rasterHash() === hash);
  T.saveLocal(); check('the scene is kept in local storage', !!store.get(T.STORE_KEY));
  store.set(T.STORE_KEY, '{"app":"flume-interface","domain":{"L":-1}}');
  check('a damaged saved scene is thrown away', T.loadLocal() === null && !store.has(T.STORE_KEY));

  // 9. resolution and scale
  const nxNormal = T.GRID.nx;
  T.setDomain({ res: 'fine' }, false);
  check('fine resolution: 1 cm cells, model given the new grid', T.GRID.nx === 2000 && C.grid.nx === 2000 && C.raster.nx === 2000);
  const cntSolid = (r) => { let s = 0; for (let c = 0; c < r.mat.length; c++) if (r.mat[c] === 1) s++; return s * r.dx * r.dx; };
  const aFine = cntSolid(T.RASTER); T.setDomain({ res: 'normal' }, false); const aNorm = cntSolid(T.RASTER);
  check('the same shapes cover the same area at another resolution (within 5%)', T.GRID.nx === nxNormal && Math.abs(aFine - aNorm) / aFine < 0.05, aFine.toFixed(4) + ' vs ' + aNorm.toFixed(4));
  const sx = T.scene.shapes[0].pts[0][0] / 20;
  T.setScale('field');
  check('switching to field scale tells the model and stretches the scene', C.setScale.includes('field') && T.DOMAIN.L === 400 && Math.abs(T.scene.shapes[0].pts[0][0] / 400 - sx) < 1e-9 && Math.abs(T.GRID.dx - 0.28) < 0.01);
  T.setScale('lab');
  check('and back to lab', T.DOMAIN.L === 20 && Math.abs(T.scene.shapes[0].pts[0][0] / 20 - sx) < 1e-9);

  // 10. running
  const fr = C.framesRunning;
  T.onKey({ key: ' ', target: {}, preventDefault() {} });
  check('Space starts the run exactly once', T.running === true);
  T.frame(1000); T.frame(1016);
  check('a running frame steps and draws the model, and the site frame runs', C.framesRunning === fr + 2 && C.render > 0 && T.backends[0].frames >= 2);
  T.onKey({ key: ' ', target: {}, preventDefault() {} });
  check('Space again pauses', T.running === false);
  const rs = C.reset; T.onKey({ key: 'r', target: {}, preventDefault() {} });
  check('R restarts the model', C.reset === rs + 1);
  check('keys typed into a number box are not taken as shortcuts', (T.onKey({ key: ' ', target: { tagName: 'INPUT' }, preventDefault() {} }), T.running === false));
  check('the render view carries the device pixel ratio', C.view && C.view.dpr === 1 && C.view.cw === W);

  // 11. gauges
  T.setTool('gauge');
  const ng = run('scene.gauges.length'), [Xg, Yg] = T.w2s(5, 0.5);
  T.onDown({ offsetX: Xg, offsetY: Yg, button: 0 });
  check('the gauge tool places a gauge where clicked', run('scene.gauges.length') === ng + 1 && Math.abs(run('scene.gauges[scene.gauges.length - 1].x') - 5) < 0.05);
  T.onDown({ offsetX: Xg + 2, offsetY: Yg + 30, button: 0 }); T.onMove({ offsetX: T.w2s(7, 0.5)[0], offsetY: Yg }); T.onUp({});
  check('a gauge can be dragged along the flume, not duplicated', run('scene.gauges.length') === ng + 1 && Math.abs(run('scene.gauges[scene.gauges.length - 1].x') - 7) < 0.05);
  T.onDown({ offsetX: Xg, offsetY: Yg, button: 0 });     // a second gauge at 5 m
  // feed a sine wave (H = 4 cm, T = 2 s) at both gauges, the second at half height
  const nx = 100, eta = new Float32Array(nx);
  for (let n = 0; n <= 400; n++) {
    const t = n * 0.05;
    for (let i = 0; i < nx; i++) { const x = (i + 0.5) * 0.2; eta[i] = 0.4 + (x < 6 ? 0.02 : 0.01) * Math.sin(2 * Math.PI * t / 2); }
    C.sample = { t, eta: Float32Array.from(eta), dx: 0.2 };
    run('sampleGauges()');
  }
  run('frame(5000)');
  const legend = getEl('gauge-legend').innerHTML;
  check('the model is asked to watch the surface while gauges exist', C.watch[C.watch.length - 1] === true);
  check('the panel shows each gauge with its mean level and wave height', /G1<\/b> at 7 m · mean 0\.4 m · H 2 cm/.test(legend) && /G2<\/b> at 5 m · mean 0\.4 m · H 4 cm/.test(legend), legend.replace(/<[^>]+>/g, '').slice(0, 160));
  check('and the wave height relative to the first gauge (transmission)', /×2\.00/.test(legend));
  const saved = T.sceneFileText();
  check('gauges are saved with the scene and come back', /"gauges"/.test(saved) && T.openSceneText(saved) && run('scene.gauges.length') === ng + 2);
  check('a right-click removes a gauge', (T.setTool('gauge'), T.onDown({ offsetX: T.w2s(5, 0.5)[0], offsetY: Yg, button: 2 }), run('scene.gauges.length') === ng + 1));

  // 12. the moving bed: the model's bed is drawn, restart goes back to the design
  const R2 = T.RASTER, N2 = R2.mat.length, h0 = T.rasterHash();
  const mat2 = Uint8Array.from(R2.mat); let flipped = -1;
  for (let c = 0; c < N2; c++) if (mat2[c] === 0) { mat2[c] = 4; flipped = c; break; }
  const comp2 = new Float32Array(3 * N2); comp2[3 * flipped + 2] = 1; const tdep2 = new Float32Array(N2).fill(-1); tdep2[flipped] = 300;
  C.api.bedChanged({ mat: mat2, comp: comp2, tdep: tdep2 });
  check('a bed moved by the model is drawn (cells, mix and deposit time reach the raster)', T.RASTER.mat[flipped] === 4 && T.RASTER.comp[3 * flipped + 2] === 1 && T.RASTER.tdep[flipped] === 300);
  const sr = C.setRaster, rs2 = C.reset;
  T.onKey({ key: 'r', target: {}, preventDefault() {} });
  check('restart goes back to the drawn design and gives the model that bed', T.rasterHash() === h0 && !T.RASTER.comp && C.setRaster === sr + 1 && C.reset === rs2 + 1);

  // 13. stage 5: the dye tool, clear dye, and a still image
  T.setTool('dye');
  const [Xd, Yd] = T.w2s(6, 0.4);
  T.onDown({ offsetX: Xd, offsetY: Yd, button: 0 }); T.onMove({ offsetX: Xd + 30, offsetY: Yd }); T.onUp({});
  check('the dye tool releases dye where clicked and along a drag', C.dye.length >= 2 && Math.abs(C.dye[0][0] - 6) < 0.05 && Math.abs(C.dye[0][1] - 0.4) < 0.02 && C.dye[0][2] > 0);
  getEl('clear-dye').dispatch('click');
  check('clear dye asks the model to clear it', C.dyeCleared === 1);
  getEl('still').dispatch('click'); run('frame(9000)');
  check('save image composites the view and downloads a PNG', downloads.includes('flume.png'));
  const mi = run('({ w: matCanvas.width, h: matCanvas.height, ux: matUX, uz: matUZ })');
  check('the material image is drawn finer than the grid (smooth outlines)', mi.w === T.GRID.nx * mi.ux && mi.h === T.GRID.nz * mi.uz && mi.ux >= 2 && mi.uz >= 2, JSON.stringify(mi));

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
})();
