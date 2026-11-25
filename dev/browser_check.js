// Load the page in a real Chrome, check it boots with WebGPU and without errors, build a demo
// scene, run it, and take a screenshot.
//   PLAYWRIGHT_CORE=<path to playwright-core> CHROME=<chrome binary> node dev/browser_check.js URL [out.png] [seconds]
// Serve the folder over http on 127.0.0.1 first (WebGPU needs a secure context; localhost is one).
'use strict';
const { chromium } = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const url = process.argv[2], out = process.argv[3] || 'flume.png', secs = +(process.argv[4] || 8);
const scale = +(process.env.DPR || 1);

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME || undefined, ignoreDefaultArgs: ['--headless'],
    args: ['--headless=new', '--enable-unsafe-webgpu', '--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan', '--use-vulkan=native', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1500, height: 720 }, deviceScaleFactor: scale });
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); else if (m.type() === 'warning' && /flume/.test(m.text())) console.log('warn: ' + m.text()); });
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(url);
  await page.waitForFunction(() => typeof booted !== 'undefined' && booted, null, { timeout: 60000 });
  const info = await page.evaluate(() => ({ banner: document.getElementById('banner').hidden ? null : document.getElementById('banner').textContent, gpu: !!(model && model.ok) }));
  console.log('booted; WebGPU model running:', info.gpu, info.banner ? '; banner: ' + info.banner : '');
  if (process.env.DEMO !== '0') {
    await page.evaluate(() => {
      // demo: wave flume, draft cells, a sloping beach, a porous reef, a solid sill, a patch of plants
      localStorage.clear();
      setDomain({ res: 'draft' }, false);
      model.setState({ type: 'wave', h0: 0.45, H: 0.07, T: 1.8 }); model.syncControls();
      scene.shapes = []; scene.bed = { pts: [[0, 0.08], [9, 0.08], [17.5, 0.62], [20, 0.66]], cls: 0 };
      curMat = 2; addShape('ellipse', [[5.0, 0.02], [7.2, 0.36]]);
      curMat = 1; addShape('poly', [[10.2, 0.2], [10.9, 0.2], [10.75, 0.42], [10.35, 0.42]]);
      curMat = 3; addShape('rect', [[14.2, 0.3], [16.4, 0.62]]);
      curMat = 4; curProps[4].cls = 2; addShape('stroke', [[11.6, 0.29], [13.6, 0.42]], 0.05);
      scene.gauges = [{ id: 1, x: 3.0 }, { id: 2, x: 12.5 }]; nextGaugeId = 3;
      model.reset(); setRunning(true);
    });
    await page.waitForTimeout(secs * 1000);
  }
  const stats = await page.evaluate(() => (model.readouts ? model.readouts() : []).map(r => r.join(': ')).join(' | '));
  console.log(stats);
  await page.screenshot({ path: out });
  console.log('screenshot', out);
  console.log(errors.length ? 'ERRORS:\n' + errors.join('\n') : 'no console errors');
  await browser.close();
  process.exit(errors.length ? 1 : 0);
})();
