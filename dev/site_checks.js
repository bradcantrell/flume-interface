// Checks for the site layer (site.js): the OpenFOAM case writer and its ZIP.
//   node dev/site_checks.js
// Running the case in OpenFOAM is dev/openfoam_case.js + dev/openfoam_compare.js.
'use strict';
const fs = require('fs'), path = require('path'), vm = require('vm'), os = require('os'), cp = require('child_process');
let backend = null;
global.registerBackend = (b) => { backend = b; };
vm.runInThisContext(fs.readFileSync(path.join(__dirname, '..', 'site.js'), 'utf8') + '\n;globalThis.FLUME_CASE = FLUME_CASE;');
const { CASES, raster } = require('./openfoam_case.js');

let pass = 0, fail = 0;
const check = (name, ok, detail) => { if (ok) { pass++; console.log('  ok  ' + name + (detail ? '  (' + detail + ')' : '')); } else { fail++; console.log('  FAIL ' + name + (detail ? '  — ' + detail : '')); } };
const listOf = (text) => { const m = text.match(/\n(\d+)\n\(\n([\s\S]*?)\)/); return m ? { n: +m[1], items: m[2].trim().split(/\s+/).filter(Boolean).map(Number) } : null; };
console.log('flume site checks');

check('the site registers a backend with an export button', backend && backend.id === 'flume-site' && typeof backend.mount === 'function');

// river case with a porous block and a plant patch
const dx = 0.02, C = { L: 6, H: 0.4, paint: (x, z) => (z < 0.04 ? 4 : (x > 2 && x < 2.5 && z < 0.2) ? 1 : (x > 3 && x < 3.6 && z < 0.3) ? 2 : (x > 4 && x < 5 && z < 0.25) ? 3 : 0) };
const r = raster(C, dx);
for (let c = 0; c < r.mat.length; c++) { if (r.mat[c] === 2) { r.p1[c] = 0.45; r.p2[c] = 0.03; } if (r.mat[c] === 3) { r.p1[c] = 12; r.p2[c] = 1.2; r.p3[c] = 0.006; } }
const p = { type: 'river', h0: 0.3, q: 0.02, hout: 0.25 };
const F = FLUME_CASE.files({ grid: { nx: r.nx, nz: r.nz, dx }, raster: r, p, sx0: 5, endTime: 30, scale: 'lab', sceneText: '{"app":"flume-interface"}', gauges: [1.5, 4.25] });
const need = ['Allrun', 'README.md', 'scene.json', 'system/blockMeshDict', 'system/controlDict', 'system/fvSchemes', 'system/fvSolution', 'system/setFieldsDict', 'system/fluidCells', 'system/cellZones',
  'constant/g', 'constant/fvModels', 'constant/momentumTransport', 'constant/phaseProperties', 'constant/physicalProperties.water', 'constant/physicalProperties.air', '0/U', '0/p_rgh', '0/alpha.water', '0/k', '0/epsilon', '0/nut'];
const missing = need.filter(f => !(f in F));
check('the river case has every file it needs', missing.length === 0, missing.join(', '));
check('blockMesh has the browser grid, one cell thick', F['system/blockMeshDict'].includes(`(${r.nx} ${r.nz} 1)`));
let nFluid = 0; for (const m of r.mat) if (m !== 1 && m !== 4) nFluid++;
const fl = listOf(F['system/fluidCells']);
check('the fluid cell set lists exactly the water, porous and plant cells', fl && fl.n === nFluid && fl.items.length === nFluid && fl.items.every(c => r.mat[c] !== 1 && r.mat[c] !== 4), fl ? fl.n + ' of ' + nFluid : 'unreadable');
// zones: labels are positions in the fluid list (subsetMesh keeps cell order)
const zt = F['system/cellZones'], zoneRe = /(\w+)\n\{\n    type cellZone;\ncellLabels      List<label> (\d+)\n\(\n([\s\S]*?)\)\n;/g;
const zonesFound = [...zt.matchAll(zoneRe)].map(m => ({ name: m[1], n: +m[2], labels: m[3].trim().split(/\s+/).map(Number) }));
const fluidList = fl.items;
const porousZ = zonesFound.find(z => z.name.startsWith('porous')), plantZ = zonesFound.find(z => z.name.startsWith('plants'));
check('one porous zone and one plant zone', zonesFound.length === 2 && porousZ && plantZ, zonesFound.map(z => z.name + ' ' + z.n).join(', '));
check('zone labels are subset-mesh labels that point back at porous and plant cells', porousZ && porousZ.labels.every(l => r.mat[fluidList[l]] === 2) && plantZ.labels.every(l => r.mat[fluidList[l]] === 3));
check('the zone file is a cellZoneList', /class\s+cellZoneList;/.test(zt));
// porosity coefficients equal the model's: d = A/nu, f = 2B; plants f = (Cd a, 0, 0)
const k = FLUME_CASE.porousCoeffs(0.45, 0.03), fv = F['constant/fvModels'];
const dP = +fv.match(/porous\d[\s\S]*?d\s+\(([^ )]+)/)[1], fP = +fv.match(/porous\d[\s\S]*?f\s+\(([^ )]+)/)[1];
const A = 1000 * 0.55 * 0.55 / 0.45 ** 3 * 1e-6 / 0.03 ** 2, B = 1.1 * 0.55 / 0.45 ** 3 / 0.03;
check('porous zone: Darcy d = A/nu and Forchheimer f = 2B, the model\'s van Gent coefficients', Math.abs(dP - A / 1e-6) / (A / 1e-6) < 1e-5 && Math.abs(fP - 2 * B) / (2 * B) < 1e-5 && Math.abs(k.A - A) < 1e-9, `d ${dP.toExponential(3)}, f ${fP.toFixed(2)}`);
const fPl = fv.match(/plants\d[\s\S]*?f\s+\(([^)]+)\)/)[1].trim().split(/\s+/).map(Number);
check('plant zone: Forchheimer on horizontal flow only, f = (Cd a, 0, 0)', Math.abs(fPl[0] - 14.4) < 1e-6 && fPl[1] === 0 && fPl[2] === 0, fPl.join(' '));
check('river inlet carries q times the cell thickness, outlet settles at the tailwater depth', F['0/U'].includes('variableHeightFlowRateInletVelocity') && F['0/U'].includes(`flowRate        ${0.02 * dx}`) && F['0/U'].includes(`UnMean          ${+(0.02 / 0.25).toPrecision(7)}`));
check('initial water fills to the water level', F['system/setFieldsDict'].includes('(1e300 0.3 1e300)'));
check('bed roughness is 2.5 d50 of the bed sediment', F['0/nut'].includes('Ks              uniform ' + 2.5 * 0.001), F['0/nut'].match(/Ks\s+uniform ([^;]+)/)[1]);
check('Allrun cuts the solids out, installs the zones, sets the water, and runs', /blockMesh[\s\S]*subsetMesh -cellSet fluid -patch walls[\s\S]*cellZones[\s\S]*setFields[\s\S]*foamRun/.test(F['Allrun']));

// wave case
const rw = raster(CASES.wave, dx), Fw = FLUME_CASE.files({ grid: { nx: rw.nx, nz: rw.nz, dx }, raster: rw, p: CASES.wave.p, sx0: 16, endTime: 20, scale: 'lab' });
check('gauges become their own OpenFOAM probe set, in order', /gauges\s*\{[\s\S]*?locations\s+\(\(1\.5 0 0\.01\) \(4\.25 0 0\.01\)\)/.test(F['system/controlDict']) && /postProcessing\/gauges/.test(F['README.md']));
check('wave case: Airy waves at the inlet with the asked height and period, closed far end', Fw['0/U'].includes('waveVelocity') && /amplitude\s+table \(\(0 0\) \(3\.2 0\.03\)\);/.test(Fw['constant/waveProperties']) && /scale\s+constant 1;/.test(Fw['constant/waveProperties']) && /period\s+1\.6;/.test(Fw['constant/waveProperties']) && /outlet\s*\{\s*type wall;/.test(Fw['system/blockMeshDict']));
check('wave case: damping over the sponge, from x = 16 m', /isotropicDamping[\s\S]*origin\s+\(16 0 0\)/.test(Fw['constant/fvModels']));
check('turbulence: k-epsilon by default', /kEpsilon/.test(F['constant/momentumTransport']) && /kEpsilon/.test(Fw['constant/momentumTransport']));
const Fl = FLUME_CASE.files({ grid: { nx: rw.nx, nz: rw.nz, dx }, raster: rw, p: CASES.wave.p, sx0: 16, endTime: 20, scale: 'lab', turbulence: 'laminar' });
check('turbulence: laminar when asked, without k, epsilon and nut fields', /simulationType\s+laminar;/.test(Fl['constant/momentumTransport']) && !('0/k' in Fl) && !('0/nut' in Fl) && !('0/epsilon' in Fl));
check('wave case: no zone file when nothing is porous or planted', !('system/cellZones' in Fw) && !/cellZones/.test(Fw['Allrun']));

// the ZIP opens and Allrun is executable
const blob = FLUME_CASE.zip(F, 'flume-case');
blob.arrayBuffer().then((ab) => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'flume-zip-')), zf = path.join(tmp, 'case.zip');
  fs.writeFileSync(zf, Buffer.from(ab));
  let ok = false, detail = '';
  try {
    const out = cp.execFileSync('python3', ['-I', '-c', `import zipfile,sys
z=zipfile.ZipFile(sys.argv[1]); bad=z.testzip(); names=z.namelist()
info=z.getinfo('flume-case/Allrun'); mode=(info.external_attr>>16)&0o777
print(len(names), bad, oct(mode), z.read('flume-case/system/blockMeshDict')[:9].decode())`, zf]).toString().trim();
    detail = out; const [n, bad, mode] = out.split(' ');
    ok = +n === Object.keys(F).length && bad === 'None' && mode === '0o755';
  } catch (e) { detail = e.message; }
  fs.rmSync(tmp, { recursive: true, force: true });
  check('the ZIP opens, every file is intact, and Allrun is executable', ok, detail);
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
});
