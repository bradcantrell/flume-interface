// Site layer for this repo: what it does with the scene.
// It exports the drawn flume as a 2D OpenFOAM case (OpenFOAM 12 or later, foamRun with the
// incompressibleVoF solver module), packed in a ZIP with scene.json and a README.
//
// flumeCaseFiles(o) is pure (no page state), so it can be checked and run outside the browser:
//   o = { grid: { nx, nz, dx }, raster, p: model params, sx0 (wave sponge start, m), endTime (s),
//         scale: 'lab' | 'field', sceneText }
// It returns { 'path/in/case': 'text', ... }.

const FLUME_CASE = (() => {
  const NU = 1e-6, ALPHA = 1000, BETA = 1.1;               // as model.js (van Gent)
  const GRAIN_D = [0.0002, 0.001, 0.01];                     // as the interface's GRAIN
  const head = (cls, obj, loc) => `/*--------------------------------*- C++ -*----------------------------------*\\
  Written by the Flume Interface. OpenFOAM 12 or later (foamRun, incompressibleVoF).
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       ${cls};
    location    "${loc}";
    object      ${obj};
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

`;
  const num = (v) => (Math.abs(v) < 1e-12 ? '0' : (+v.toPrecision(7)).toString());
  const vec = (a) => '(' + a.map(num).join(' ') + ')';

  // bed height at a column: solid cells counted up from the floor (as the model's bedAt)
  function bedAt(r, i) { let k = 0; while (k < r.nz && (r.mat[i + r.nx * k] === 1 || r.mat[i + r.nx * k] === 4)) k++; return k * r.dx; }

  function porousCoeffs(n, d) {
    n = Math.min(Math.max(n || 0.4, 0.05), 0.95); d = Math.max(d || 0.04, 1e-4);
    const A = ALPHA * (1 - n) * (1 - n) / (n * n * n) * NU / (d * d), B = BETA * (1 - n) / (n * n * n) / d;
    return { A, B };
  }

  function files(o) {
    const { grid, raster: r, p } = o, { nx, nz, dx } = grid, L = nx * dx, H = nz * dx, t = dx, N = nx * nz;
    const river = p.type !== 'wave', zIn = bedAt(r, 0), zOut = bedAt(r, nx - 1);
    const F = {};

    // ── mesh: one block, then subsetMesh keeps the water cells ──
    F['system/blockMeshDict'] = head('dictionary', 'blockMeshDict', 'system') + `convertToMeters 1;

vertices
(
    (0 0 0) (${num(L)} 0 0) (${num(L)} ${num(H)} 0) (0 ${num(H)} 0)
    (0 0 ${num(t)}) (${num(L)} 0 ${num(t)}) (${num(L)} ${num(H)} ${num(t)}) (0 ${num(H)} ${num(t)})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (${nx} ${nz} 1) simpleGrading (1 1 1)
);

boundary
(
    inlet       { type patch; faces ((0 4 7 3)); }
    outlet      { type ${river ? 'patch' : 'wall'}; faces ((1 2 6 5)); }
    lowerWall   { type wall;  faces ((0 1 5 4)); }
    atmosphere  { type patch; faces ((3 7 6 2)); }
    walls       { type wall;  faces (); }
    frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); }
);
`;
    // blockMesh numbers cells x fastest, then y: label = i + nx * k, as the raster
    const fluid = [], newLabel = new Int32Array(N).fill(-1);
    for (let c = 0; c < N; c++) if (r.mat[c] !== 1 && r.mat[c] !== 4) { newLabel[c] = fluid.length; fluid.push(c); }
    F['system/fluidCells'] = head('cellSet', 'fluid', 'constant/polyMesh/sets') + fluid.length + '\n(\n' + chunk(fluid) + ')\n';

    // ── porous and plant zones: one per distinct set of properties, in subset-mesh labels ──
    const groups = new Map();
    for (let c = 0; c < N; c++) {
      const m = r.mat[c]; if (m !== 2 && m !== 3) continue;
      const key = m + ':' + (+r.p1[c].toPrecision(5)) + ':' + (+r.p2[c].toPrecision(5));
      if (!groups.has(key)) groups.set(key, { m, p1: r.p1[c], p2: r.p2[c], cells: [] });
      groups.get(key).cells.push(newLabel[c]);
    }
    const zones = [...groups.values()].map((g, j) => Object.assign(g, { name: (g.m === 2 ? 'porous' : 'plants') + (j + 1) }));
    if (zones.length) {
      F['system/cellZones'] = head('cellZoneList', 'cellZones', 'constant/polyMesh') + zones.length + '\n(\n' +
        zones.map(z => `${z.name}\n{\n    type cellZone;\ncellLabels      List<label> ${z.cells.length}\n(\n${chunk(z.cells)})\n;\n}\n`).join('\n') + ')\n';
    }

    // ── physics ──
    F['constant/g'] = head('uniformDimensionedVectorField', 'g', 'constant') + 'dimensions      [0 1 -2 0 0 0 0];\nvalue           (0 -9.81 0);\n';
    F['constant/phaseProperties'] = head('dictionary', 'phaseProperties', 'constant') + 'phases          (water air);\n\nsigma           0.07;\n';
    F['constant/physicalProperties.water'] = head('dictionary', 'physicalProperties.water', 'constant') + 'viscosityModel  constant;\n\nnu              1e-06;\n\nrho             1000;\n';
    F['constant/physicalProperties.air'] = head('dictionary', 'physicalProperties.air', 'constant') + 'viscosityModel  constant;\n\nnu              1.48e-05;\n\nrho             1;\n';
    // turbulence: k-epsilon with rough walls, or laminar (o.turbulence). OpenFOAM's wave tutorial runs
    // laminar, and k-epsilon damps surface waves (20 to 30% over 10 m in a test); but a laminar run
    // with waves breaking over a bar never finished (its time step collapsed).
    const laminar = o.turbulence === 'laminar';
    F['constant/momentumTransport'] = head('dictionary', 'momentumTransport', 'constant') + (laminar ? 'simulationType  laminar;\n'
      : 'simulationType  RAS;\n\nRAS\n{\n    model           kEpsilon;\n    turbulence      on;\n    printCoeffs     on;\n}\n');

    // porosityForce: S = -(mu d + rho |U| f / 2) U, so d = A / nu and f = 2 B (model's A, B)
    let fvModels = '';
    for (const z of zones) {
      let d = [0, 0, 0], f = [0, 0, 0], what;
      if (z.m === 2) { const k = porousCoeffs(z.p1, z.p2); d = [k.A / NU, k.A / NU, k.A / NU]; f = [2 * k.B, 2 * k.B, 2 * k.B]; what = `porous: porosity ${num(z.p1)}, stone ${num(z.p2)} m (van Gent coefficients ${ALPHA}, ${BETA})`; }
      else { f = [(z.p2 || 1) * (z.p1 || 8), 0, 0]; what = `plants: frontal area ${num(z.p1)} 1/m, drag coefficient ${num(z.p2)}, drag on horizontal flow only`; }
      fvModels += `// ${what}
${z.name}
{
    type            porosityForce;

    porosityForce
    {
        cellZone        ${z.name};
        type            DarcyForchheimer;
        d   ${vec(d)};
        f   ${vec(f)};
        coordinateSystem
        {
            type    cartesian;
            origin  (0 0 0);
            coordinateRotation { type axesRotation; e1 (1 0 0); e2 (0 1 0); }
        }
    }
}

`;
    }
    // wave flume: damp the far end over the same length as the browser's sponge
    let wave = null;
    if (!river) {
      const h = Math.max(p.h0 - zIn, dx), T = p.T || 2, Hw = p.H || 0.05;
      wave = { h, T, Hw };
      const Ls = Math.max(L - o.sx0, dx);
      fvModels += `// wave flume: waves are absorbed over the last ${num(Ls)} m (the browser's sponge)
damping
{
    type            isotropicDamping;
    origin          (${num(o.sx0)} 0 0);
    direction       (1 0 0);
    scale
    {
        type        halfCosineRamp;
        start       0;
        duration    ${num(Ls)};
    }
    value           (0 0 0);
    lambda          ${num(2 / T)};
}

`;
      F['constant/waveProperties'] = head('dictionary', 'waveProperties', 'constant') + `origin          (0 ${num(p.h0)} 0);

direction       (1 0 0);

waves
(
    Airy
    {
        depth       ${num(h)};
        period      ${num(T)};
        // ramped in over two periods, as the browser's paddle
        amplitude   table ((0 0) (${num(2 * T)} ${num(Hw / 2)}));
        phase       0;
        angle       0;
    }
);

UMean           (0 0 0);

// scale is a function of position along the direction (1 everywhere here)
scale           constant 1;

crossScale      constant 1;
`;
    }
    if (fvModels) F['constant/fvModels'] = head('dictionary', 'fvModels', 'constant') + fvModels;

    // ── initial and boundary fields ──
    const depth = Math.max(p.h0 - zIn, dx), Uref = river ? Math.max(p.q / depth, 0.01) : 0.05;
    const kIn = 1.5 * (0.05 * Uref) ** 2, epsIn = Math.pow(0.09, 0.75) * Math.pow(kIn, 1.5) / (0.07 * depth);
    const ksBed = 2.5 * (GRAIN_D[r.cls[nx * Math.max(0, Math.round(zIn / dx) - 1)]] || GRAIN_D[0]);
    const flowRate = river ? p.q * t : 0;
    const hOutDepth = Math.max(p.hout || depth, dx);
    const pat = (o2) => Object.entries(o2).map(([k, v]) => `    ${k}\n    {\n${v.split('\n').map(l => '        ' + l).join('\n')}\n    }`).join('\n');
    const field = (cls, obj, dim, internal, bf) => head(cls, obj, '0') + `dimensions      ${dim};\n\ninternalField   ${internal};\n\nboundaryField\n{\n${pat(bf)}\n    frontAndBack\n    {\n        type            empty;\n    }\n}\n`;
    const wall = { U: 'type            noSlip;', p: 'type            fixedFluxPressure;', a: 'type            zeroGradient;',
      k: `type            kqRWallFunction;\nvalue           uniform ${num(kIn)};`, e: `type            epsilonWallFunction;\nvalue           uniform ${num(epsIn)};`,
      nut: `type            nutkRoughWallFunction;\nKs              uniform ${num(ksBed)};\nCs              uniform 0.5;\nvalue           uniform 0;` };
    const inU = river ? `type            variableHeightFlowRateInletVelocity;\nflowRate        ${num(flowRate)};\nalpha           alpha.water;\nvalue           uniform (0 0 0);` : 'type            waveVelocity;\nlibs            ("libwaves.so");\nvalue           uniform (0 0 0);';
    const inA = river ? 'type            variableHeightFlowRate;\nlowerBound      0;\nupperBound      1;\nvalue           uniform 0;' : 'type            waveAlpha;\nlibs            ("libwaves.so");\ninletOutlet     true;\nvalue           uniform 0;';
    const outU = river ? `type            outletPhaseMeanVelocity;\nUnMean          ${num(p.q / hOutDepth)};\nalpha           alpha.water;\nvalue           uniform (0 0 0);` : wall.U;
    F['0/U'] = field('volVectorField', 'U', '[0 1 -1 0 0 0 0]', 'uniform (0 0 0)', { inlet: inU, outlet: outU, lowerWall: wall.U, walls: wall.U, atmosphere: 'type            pressureInletOutletVelocity;\nvalue           uniform (0 0 0);' });
    F['0/alpha.water'] = field('volScalarField', 'alpha.water', '[0 0 0 0 0 0 0]', 'uniform 0', { inlet: inA, outlet: river ? 'type            zeroGradient;' : wall.a, lowerWall: wall.a, walls: wall.a, atmosphere: 'type            inletOutlet;\ninletValue      uniform 0;\nvalue           uniform 0;' });
    F['0/p_rgh'] = field('volScalarField', 'p_rgh', '[1 -1 -2 0 0 0 0]', 'uniform 0', { inlet: river ? 'type            zeroGradient;' : wall.p, outlet: river ? 'type            zeroGradient;' : wall.p, lowerWall: wall.p, walls: wall.p, atmosphere: 'type            prghTotalPressure;\np0              uniform 0;\nvalue           uniform 0;' });
    const free = (v) => `type            inletOutlet;\ninletValue      uniform ${num(v)};\nvalue           uniform ${num(v)};`;
    if (!laminar) F['0/k'] = field('volScalarField', 'k', '[0 2 -2 0 0 0 0]', 'uniform ' + num(kIn), { inlet: river ? `type            fixedValue;\nvalue           uniform ${num(kIn)};` : free(kIn), outlet: river ? free(kIn) : wall.k, lowerWall: wall.k, walls: wall.k, atmosphere: free(kIn) });
    if (!laminar) F['0/epsilon'] = field('volScalarField', 'epsilon', '[0 2 -3 0 0 0 0]', 'uniform ' + num(epsIn), { inlet: river ? `type            fixedValue;\nvalue           uniform ${num(epsIn)};` : free(epsIn), outlet: river ? free(epsIn) : wall.e, lowerWall: wall.e, walls: wall.e, atmosphere: free(epsIn) });
    if (!laminar) F['0/nut'] = field('volScalarField', 'nut', '[0 2 -1 0 0 0 0]', 'uniform 0', { inlet: 'type            calculated;\nvalue           uniform 0;', outlet: river ? 'type            calculated;\nvalue           uniform 0;' : wall.nut, lowerWall: wall.nut, walls: wall.nut, atmosphere: 'type            calculated;\nvalue           uniform 0;' });

    F['system/setFieldsDict'] = head('dictionary', 'setFieldsDict', 'system') + `defaultValues
{
    alpha.water 0;
}

zones
{
    water
    {
        type        box;
        box         (-1e300 -1e300 -1e300) (1e300 ${num(p.h0)} 1e300);
        values
        {
            alpha.water 1;
        }
    }
}
`;

    // ── run control ──
    const probes = []; for (let j = 1; j < 20; j++) probes.push(`(${num(L * j / 20)} 0 ${num(t / 2)})`);
    F['system/controlDict'] = head('dictionary', 'controlDict', 'system') + `solver          incompressibleVoF;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         ${num(o.endTime)};
deltaT          ${num(Math.min(0.2 * dx, 0.001))};
writeControl    adjustableRunTime;
writeInterval   ${num(Math.max(o.endTime / 10, 0.1))};
purgeWrite      0;
writeFormat     binary;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable yes;
adjustTimeStep  yes;
maxCo           ${river ? '0.5' : '0.3'};
maxAlphaCo      ${river ? '0.5' : '0.3'};
maxDeltaT       1;

functions
{
    // water surface height at 19 points along the flume, every time step
    surface
    {
        type            interfaceHeight;
        libs            ("libfieldFunctionObjects.so");
        locations       (${probes.join(' ')});
        alpha           alpha.water;
    }${(o.gauges || []).length ? `

    // the gauges placed in the browser, in their order (G1, G2, ...)
    gauges
    {
        type            interfaceHeight;
        libs            ("libfieldFunctionObjects.so");
        locations       (${o.gauges.map(x => `(${num(x)} 0 ${num(t / 2)})`).join(' ')});
        alpha           alpha.water;
    }` : ''}
}
`;
    F['system/fvSchemes'] = head('dictionary', 'fvSchemes', 'system') + `ddtSchemes      { default Euler; }

gradSchemes     { default Gauss linear; }

divSchemes
{
    div(rhoPhi,U)   Gauss linearUpwindV grad(U);
    div(phi,alpha)  Gauss interfaceCompression vanLeer 1;
    div(phi,k)      Gauss upwind;
    div(phi,epsilon) Gauss upwind;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes { default Gauss linear corrected; }

interpolationSchemes { default linear; }

snGradSchemes   { default corrected; }
`;
    F['system/fvSolution'] = head('dictionary', 'fvSolution', 'system') + `solvers
{
    "alpha.water.*"
    {
${river ? `        nCorrectors     1;
        nSubCycles      polynomial (0 4);` : `        // as OpenFOAM's wave tutorial: semi-implicit MULES keeps steep waves bounded
        nCorrectors     2;
        nSubCycles      1;
        MULESCorr       yes;
        alphaApplyPrevCorr yes;
        MULES
        {
            nIter       3;
        }
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0;
        minIter         1;`}
    }
    "pcorr.*"
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-10;
        relTol          0;
    }
    p_rgh
    {
${river ? `        solver          GAMG;
        smoother        DIC;` : `        // PCG, as the weirOverflow tutorial: GAMG stopped with a division by zero in a wave test
        solver          PCG;
        preconditioner  DIC;`}
        tolerance       1e-07;
        relTol          0.05;
    }
    p_rghFinal
    {
        $p_rgh;
        relTol          0;
    }
    "(U|k|epsilon)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }
    "(U|k|epsilon)Final"
    {
        $U;
        relTol          0;
    }
}

PIMPLE
{
    momentumPredictor ${river ? 'no' : 'yes'};
    nCorrectors     3;
    nNonOrthogonalCorrectors 0;
}
`;
    F['Allrun'] = `#!/bin/sh
# Builds the mesh from the drawn flume and runs it. Needs OpenFOAM 12 or later sourced.
cd "\${0%/*}" || exit 1
. "$WM_PROJECT_DIR/bin/tools/RunFunctions"
runApplication blockMesh
mkdir -p constant/polyMesh/sets
cp system/fluidCells constant/polyMesh/sets/fluid
runApplication subsetMesh -cellSet fluid -patch walls
${zones.length ? 'cp system/cellZones constant/polyMesh/cellZones\n' : ''}runApplication setFields
runApplication foamRun
`;
    F['README.md'] = `# Flume case from the Flume Interface

A 2D OpenFOAM case of the flume you drew (${num(L)} m by ${num(H)} m, ${nx} by ${nz} cells of ${num(dx * 100)} cm,
${fluid.length} of them water or air). ${river ? `River: ${num(p.q)} m²/s per metre comes in at the left over the wet depth; the
right end lets it out and settles at a tailwater depth of ${num(hOutDepth)} m.` : `Wave flume: Airy waves of ${num(wave.Hw)} m height and ${num(wave.T)} s period
come in at the left in ${num(wave.h)} m of water; the far end is a wall with a damping zone.`}
${zones.length ? `Porous areas and plants are ${zones.length} porosity zone(s), with the same drag coefficients as the browser.` : ''}

## Run

Source OpenFOAM (12 or later), then

    ./Allrun

The water surface height at 19 points along the flume is written every step to
\`postProcessing/surface/0/height.dat\`${(o.gauges || []).length ? `, and at your ${o.gauges.length} gauge(s) (G1 first) to
\`postProcessing/gauges/0/height.dat\`` : ''}. Open the case in ParaView with \`paraFoam\` or by opening an
empty file named \`case.foam\`.

## How it matches the browser

- The mesh is the browser's grid, one cell thick. Solid and sediment cells are cut out
  (\`subsetMesh\`), and their faces become the \`walls\` patch.
- ${laminar ? 'Turbulence is off (laminar), as OpenFOAM\'s wave tutorial. If the waves break and the time step\n  collapses, export again with k-epsilon.' : `Turbulence is k-epsilon with rough wall functions (Ks = \${num(ksBed)} m, from the bed's grain size).
  k-epsilon damps surface waves somewhat.`} The browser uses a Smagorinsky eddy viscosity and log-law friction.
- Porous areas use OpenFOAM's Darcy-Forchheimer porosity with the browser's coefficients;
  plants are a Forchheimer term on horizontal flow only.
- Sediment does not move in this case. It is the flow over the starting bed.

\`scene.json\` is the drawn scene. The Flume Interface opens it again.
`;
    if (o.sceneText) F['scene.json'] = o.sceneText;
    return F;
  }
  function chunk(a) { let s = ''; for (let j = 0; j < a.length; j += 16) s += a.slice(j, j + 16).join(' ') + '\n'; return s; }

  // ── a small ZIP writer (stored, no compression) ──
  const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
  function crc32(b) { let c = 0xFFFFFFFF; for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 255] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; }
  function zip(filesObj, folder) {
    const enc = new TextEncoder(), parts = [], central = []; let off = 0;
    for (const [name0, text] of Object.entries(filesObj)) {
      const name = enc.encode(folder + '/' + name0), data = enc.encode(text), crc = crc32(data);
      const exec = name0 === 'Allrun';
      const lh = new DataView(new ArrayBuffer(30));
      lh.setUint32(0, 0x04034b50, true); lh.setUint16(4, 20, true); lh.setUint32(14, crc, true); lh.setUint32(18, data.length, true); lh.setUint32(22, data.length, true); lh.setUint16(26, name.length, true);
      parts.push(new Uint8Array(lh.buffer), name, data);
      const ch = new DataView(new ArrayBuffer(46));
      ch.setUint32(0, 0x02014b50, true); ch.setUint16(4, 0x0314, true); ch.setUint16(6, 20, true); ch.setUint32(16, crc, true); ch.setUint32(20, data.length, true); ch.setUint32(24, data.length, true);
      ch.setUint16(28, name.length, true); ch.setUint32(38, ((exec ? 0o100755 : 0o100644) << 16) >>> 0, true); ch.setUint32(42, off, true);
      central.push(new Uint8Array(ch.buffer), name);
      off += 30 + name.length + data.length;
    }
    const csize = central.reduce((s, a) => s + a.length, 0), n = Object.keys(filesObj).length;
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true); end.setUint16(8, n, true); end.setUint16(10, n, true); end.setUint32(12, csize, true); end.setUint32(16, off, true);
    return new Blob([...parts, ...central, new Uint8Array(end.buffer)], { type: 'application/zip' });
  }
  return { files, zip, porousCoeffs };
})();

if (typeof registerBackend === 'function') registerBackend({
  id: 'flume-site',
  title: 'Flume Interface',
  mount(slot) {
    slot.innerHTML = `<h2>OpenFOAM</h2>
      <div class="row"><label for="of-end">run time (s)</label><input id="of-end" type="number" min="1" step="1" value="60"></div>
      <div class="row"><label for="of-turb">turbulence</label><select id="of-turb"><option value="auto">auto</option><option value="kEpsilon">k-epsilon</option><option value="laminar">none (laminar)</option></select></div>
      <div class="btns"><button id="of-export">export case</button></div>
      <p class="note">A 2D case of this flume for OpenFOAM 12 or later, with this scene inside.</p>`;
    document.getElementById('of-export').addEventListener('click', () => {
      const end = Math.max(1, parseFloat(document.getElementById('of-end').value) || 60);
      const p = model && model.state ? model.state() : { type: 'river', h0: 0.4, q: 0.1, hout: 0.4, H: 0.08, T: 2 };
      const sx0 = model && model.P ? model.P.sx0 : 0.8 * DOMAIN.L;
      // auto: k-epsilon for a river, laminar for waves (choose k-epsilon when the waves break)
      let turbulence = document.getElementById('of-turb').value;
      if (turbulence === 'auto') turbulence = p.type === 'wave' ? 'laminar' : 'kEpsilon';
      const F = FLUME_CASE.files({ grid: GRID, raster: RASTER, p, sx0, endTime: end, scale: DOMAIN.scale, sceneText: sceneFileText(), turbulence, gauges: scene.gauges.map(g => g.x) });
      const a = document.createElement('a'); a.href = URL.createObjectURL(FLUME_CASE.zip(F, 'flume-case')); a.download = 'flume-case.zip';
      document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
    });
  },
});
