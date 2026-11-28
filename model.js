// Flume model: how the water moves, on the GPU (WebGPU).
//
// A 2D section (x along the flume, z up) on a staggered MAC grid. Each step:
//   extrapolate velocity into the air, semi-Lagrangian advection (RK2) + gravity,
//   boundaries (river inflow/outflow or wave paddle), sponge (wave flume),
//   pressure: CG preconditioned by a geometric multigrid V-cycle (red-black Gauss-Seidel),
//   projection, conservative directional-split PLIC volume of fluid (Weymouth-Yue).
//
// The interface hands the model a material raster (see MODEL EXTENSION POINT in index.html).
// Solid and sediment cells are walls (sediment moves in stage 4). Porous and plant cells are
// water with drag: Darcy-Forchheimer for porous (van Gent coefficients, superficial velocity,
// as OpenFOAM's porosityForce), 1/2 Cd a |u| u on horizontal flow for plants. Turbulence is a
// Smagorinsky eddy viscosity; walls and the bed have log-law friction with ks = 2.5 d50.
// Porous cells keep full volume for water (no porosity in storage), as in the OpenFOAM export.
// Everything here runs in a browser or in Node (Dawn).
(function () {
  'use strict';

  const G = 9.81;
  const COMB = { S: 14, M: 600, P: 10, te: 1 / 60 };
  const WIRE = { W: 4, R: 16, K: 48 };                     // bubble wires: wires, timelines kept per wire, marks per timeline     // smoke comb: release points, ring length, pulse every P, release interval (s)
  const K_CG = 8;            // CG iterations per step (8 reaches the single-precision floor, see spike)
  const CFL = 0.4;
  const NU_WATER = 1e-6;     // m2/s
  const CS_SMAG = 0.17;      // Smagorinsky constant
  const POROUS_ALPHA = 1000, POROUS_BETA = 1.1;     // van Gent (1995) coefficients
  const GRAIN_D = [0.0002, 0.001, 0.01];            // m: fine sand, coarse sand, gravel (as the interface's GRAIN)
  const SED = { s: 2.65, porosity: 0.4, repose: 32, slope: 1.5 };   // quartz; angle of repose in degrees
  // critical Shields stress (Soulsby & Whitehouse 1997) and fall velocity (Ferguson & Church 2004)
  function grainProps(d) {
    const R = SED.s - 1, Ds = d * Math.cbrt(R * G / (NU_WATER * NU_WATER));
    const thc = 0.30 / (1 + 1.2 * Ds) + 0.055 * (1 - Math.exp(-0.020 * Ds));
    const ws = R * G * d * d / (18 * NU_WATER + Math.sqrt(0.75 * 1.0 * R * G * d * d * d));
    return { thc, ws };
  }

  // ───────────────────────── WGSL ─────────────────────────
  const HEAD = `
struct P { nx:u32, nz:u32, cnx:u32, cnz:u32, dx:f32, dt:f32, g:f32, mode:f32,
           uin:f32, hin:f32, uout:f32, hout:f32, S:f32, om:f32, sx0:f32, srate:f32,
           zin:f32, zout:f32, q:f32, nu:f32, cs:f32, amp:f32, absorb:f32, pad3:f32 };
@group(0) @binding(0) var<uniform> prm: P;
`;
  const PLIC = `
fn unitArea(c1i:f32, c2i:f32, bi:f32) -> f32 {
  var c1 = c1i; var c2 = c2i; var b = bi;
  if (c1 < 0.0) { b -= c1; c1 = -c1; }
  if (c2 < 0.0) { b -= c2; c2 = -c2; }
  let s = c1 + c2;
  if (s < 1e-12) { return select(0.0, 1.0, b >= 0.0); }
  c1 /= s; c2 /= s; b /= s;
  if (b <= 0.0) { return 0.0; } if (b >= 1.0) { return 1.0; }
  let m1 = min(c1, c2); let m2 = max(c1, c2);
  if (m1 < 1e-7) { return b; }
  if (b < m1) { return b*b/(2.0*m1*m2); }
  if (b <= m2) { return (b - m1*0.5)/m2; }
  let r = 1.0 - b; return 1.0 - r*r/(2.0*m1*m2);
}
fn alphaFor(n1:f32, n2:f32, f:f32) -> f32 {
  let c1 = abs(n1); let c2 = abs(n2); let s = c1 + c2;
  let m1 = min(c1, c2)/s; let m2 = max(c1, c2)/s;
  var b: f32;
  if (m1 < 1e-7) { b = f; } else {
    let V1 = m1/(2.0*m2);
    if (f <= V1) { b = sqrt(2.0*m1*m2*f); } else if (f <= 1.0 - V1) { b = f*m2 + m1*0.5; } else { b = 1.0 - sqrt(2.0*m1*m2*(1.0 - f)); }
  }
  return b*s + min(n1, 0.0) + min(n2, 0.0);
}
fn fAt(i:i32, k:i32) -> f32 {
  let ii = u32(clamp(i, 0, i32(prm.nx) - 1)); let kk = u32(clamp(k, 0, i32(prm.nz) - 1));
  return f[ii + prm.nx*kk];
}
fn flux1(di:i32, dk:i32, vel:f32, s:f32, dir:u32) -> f32 {
  let fd = f[u32(di) + prm.nx*u32(dk)];
  if (fd <= 1e-6) { return 0.0; }
  if (fd >= 1.0 - 1e-6) { return s; }
  let gx = (fAt(di+1,dk+1) + 2.0*fAt(di+1,dk) + fAt(di+1,dk-1)) - (fAt(di-1,dk+1) + 2.0*fAt(di-1,dk) + fAt(di-1,dk-1));
  let gz = (fAt(di+1,dk+1) + 2.0*fAt(di,dk+1) + fAt(di-1,dk+1)) - (fAt(di+1,dk-1) + 2.0*fAt(di,dk-1) + fAt(di-1,dk-1));
  let n1 = -gx; let n2 = -gz;
  if (abs(n1) + abs(n2) < 1e-9) { return fd*s; }
  let al = alphaFor(n1, n2, fd);
  if (dir == 0u) {
    if (vel > 0.0) { return s*unitArea(n1*s, n2, al - n1*(1.0 - s)); }
    return s*unitArea(n1*s, n2, al);
  }
  if (vel > 0.0) { return s*unitArea(n1, n2*s, al - n2*(1.0 - s)); }
  return s*unitArea(n1, n2*s, al);
}
`;
  const SAMPLE = `
fn sU(x:f32, z:f32) -> f32 {
  let fx = clamp(x/prm.dx, 0.0, f32(prm.nx)); let fz = clamp(z/prm.dx - 0.5, 0.0, f32(prm.nz) - 1.0);
  let i = min(u32(fx), prm.nx - 1u); let k = min(u32(fz), max(prm.nz, 2u) - 2u);
  let a = fx - f32(i); let b = fz - f32(k); let W = prm.nx + 1u; let k1 = min(k + 1u, prm.nz - 1u);
  return (1.0-b)*((1.0-a)*u[i+W*k] + a*u[i+1u+W*k]) + b*((1.0-a)*u[i+W*k1] + a*u[i+1u+W*k1]);
}
fn sW(x:f32, z:f32) -> f32 {
  let fx = clamp(x/prm.dx - 0.5, 0.0, f32(prm.nx) - 1.0); let fz = clamp(z/prm.dx, 0.0, f32(prm.nz));
  let i = min(u32(fx), prm.nx - 2u); let k = min(u32(fz), prm.nz - 1u);
  let a = fx - f32(i); let b = fz - f32(k); let i1 = min(i + 1u, prm.nx - 1u); let N = prm.nx;
  return (1.0-b)*((1.0-a)*w[i+N*k] + a*w[i1+N*k]) + b*((1.0-a)*w[i+N*(k+1u)] + a*w[i1+N*(k+1u)]);
}
`;
  const SAMPLE2U = `
fn sU2(x:f32, z:f32) -> f32 {
  let fx = clamp(x/prm.dx, 0.0, f32(prm.nx)); let fz = clamp(z/prm.dx - 0.5, 0.0, f32(prm.nz) - 1.0);
  let i = min(u32(fx), prm.nx - 1u); let k = min(u32(fz), max(prm.nz, 2u) - 2u);
  let a = fx - f32(i); let b = fz - f32(k); let W = prm.nx + 1u; let k1 = min(k + 1u, prm.nz - 1u);
  return (1.0-b)*((1.0-a)*u2[i+W*k] + a*u2[i+1u+W*k]) + b*((1.0-a)*u2[i+W*k1] + a*u2[i+1u+W*k1]);
}
`;
  const SAMPLE2W = `
fn sW2(x:f32, z:f32) -> f32 {
  let fx = clamp(x/prm.dx - 0.5, 0.0, f32(prm.nx) - 1.0); let fz = clamp(z/prm.dx, 0.0, f32(prm.nz));
  let i = min(u32(fx), prm.nx - 2u); let k = min(u32(fz), prm.nz - 1u);
  let a = fx - f32(i); let b = fz - f32(k); let i1 = min(i + 1u, prm.nx - 1u); let N = prm.nx;
  return (1.0-b)*((1.0-a)*w2[i+N*k] + a*w2[i1+N*k]) + b*((1.0-a)*w2[i+N*(k+1u)] + a*w2[i1+N*(k+1u)]);
}
`;
  const NB = `
fn nbr(c:u32, i:u32, k:u32, x: ptr<function, f32>) -> f32 {
  var d = 0.0; var s = 0.0; let N = prm.nx;
  if (i > 0u)        { let t = typ[c-1u]; if (t != 0u) { d += 1.0; if (t == 1u) { s += X[c-1u]; } } }
  if (i < prm.nx-1u) { let t = typ[c+1u]; if (t != 0u) { d += 1.0; if (t == 1u) { s += X[c+1u]; } } }
  if (k > 0u)        { let t = typ[c-N];  if (t != 0u) { d += 1.0; if (t == 1u) { s += X[c-N]; } } }
  if (k < prm.nz-1u) { let t = typ[c+N];  if (t != 0u) { d += 1.0; if (t == 1u) { s += X[c+N]; } } }
  *x = s; return d;
}
`;
  // Level 0 of the pressure operator places the free surface where it is (ghost fluid): an air
  // neighbour adds 1/theta to the diagonal, theta = fraction of a cell from this centre to the
  // surface (from the volume fractions; clamped at 0.1). Coarse levels keep theta = 1.
  // The level-0 pressure operator carries two corrections:
  //  - ghost fluid: an air neighbour adds 1/theta, theta = fraction of a cell from this centre to
  //    the surface (from the volume fractions, clamped at 0.1), so the surface sits where it is;
  //  - drag inside the solve: each face has beta = 1/(1 + dt K) (K = drag + wall friction), and the
  //    projection is u = beta (u* - dt grad p), so steady flow feels the full drag at any dt.
  // Coarse multigrid levels keep theta = beta = 1 (they only precondition).
  const NBG = `
fn theta(c:u32, n:u32) -> f32 { return clamp(F[c] + F[n] - 0.5, 0.1, 1.0); }
fn nbr(c:u32, i:u32, k:u32, x: ptr<function, f32>) -> f32 {
  var d = 0.0; var s = 0.0; let N = prm.nx; let W = prm.nx + 1u;
  if (i > 0u)        { let t = typ[c-1u]; let b = BU[i+W*k];      if (t == 1u) { d += b; s += b*X[c-1u]; } else if (t == 2u) { d += b/theta(c, c-1u); } }
  if (i < prm.nx-1u) { let t = typ[c+1u]; let b = BU[i+1u+W*k];   if (t == 1u) { d += b; s += b*X[c+1u]; } else if (t == 2u) { d += b/theta(c, c+1u); } }
  if (k > 0u)        { let t = typ[c-N];  let b = BW[i+N*k];      if (t == 1u) { d += b; s += b*X[c-N]; }  else if (t == 2u) { d += b/theta(c, c-N); } }
  if (k < prm.nz-1u) { let t = typ[c+N];  let b = BW[i+N*(k+1u)]; if (t == 1u) { d += b; s += b*X[c+N]; }  else if (t == 2u) { d += b/theta(c, c+N); } }
  *x = s; return d;
}
`;
  const AIR = `fn isAir(i:i32, k:i32) -> bool { if (i < 0 || k < 0 || i >= i32(prm.nx) || k >= i32(prm.nz)) { return false; } return typ[u32(i) + prm.nx*u32(k)] == 2u; }\n`;
  const MM = `fn mm(a:f32, b:f32) -> f32 { if (a*b <= 0.0) { return 0.0; } return select(b, a, abs(a) < abs(b)); }
`;
  function K(binds, body, extra = '', wg = 64) {
    let decl = '';
    binds.forEach(([n, t], j) => {
      if (t === 'U') decl += `@group(0) @binding(${j + 1}) var<uniform> ${n}: array<vec4f, 4>;\nfn spv(i: u32) -> f32 { return ${n}[i / 4u][i % 4u]; }\n`;
      else decl += `@group(0) @binding(${j + 1}) var<storage, read_write> ${n}: array<${t === 'u' ? 'u32' : 'f32'}>;\n`;
    });
    return HEAD + decl + extra + `\n@compute @workgroup_size(${wg}) fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {\n let c = gid.x; _ = prm.nx;\n${body}\n}`;
  }
  const SRC = {
    classify: K([['f', 'f'], ['solid', 'u'], ['typ', 'u']], `if (c >= prm.nx*prm.nz) {return;}
      if (solid[c] != 0u) { typ[c] = 0u; } else if (f[c] > 0.5) { typ[c] = 1u; } else { typ[c] = 2u; }`),
    clearSolid: K([['f', 'f'], ['solid', 'u']], `if (c >= prm.nx*prm.nz) {return;} if (solid[c] != 0u) { f[c] = 0.0; }`),
    validU: K([['typ', 'u'], ['uv', 'u']], `let W = prm.nx+1u; if (c >= W*prm.nz) {return;} let i = c % W; let k = c / W;
      var l = 0u; var r = 0u; if (i > 0u) { l = typ[i-1u+prm.nx*k]; } if (i < prm.nx) { r = typ[i+prm.nx*k]; }
      uv[c] = select(0u, 1u, l == 1u || r == 1u);`),
    validW: K([['typ', 'u'], ['wv', 'u']], `let N = prm.nx; if (c >= N*(prm.nz+1u)) {return;} let i = c % N; let k = c / N;
      var b = 0u; var t = 0u; if (k > 0u) { b = typ[i+N*(k-1u)]; } if (k < prm.nz) { t = typ[i+N*k]; }
      wv[c] = select(0u, 1u, b == 1u || t == 1u);`),
    extrapU: K([['a', 'f'], ['av', 'u'], ['o', 'f'], ['ov', 'u']], `let W = prm.nx+1u; if (c >= W*prm.nz) {return;} let i = c % W; let k = c / W;
      if (av[c] == 1u) { o[c] = a[c]; ov[c] = 1u; return; }
      var s = 0.0; var n = 0.0;
      if (i > 0u && av[c-1u] == 1u) { s += a[c-1u]; n += 1.0; }
      if (i < prm.nx && av[c+1u] == 1u) { s += a[c+1u]; n += 1.0; }
      if (k > 0u && av[c-W] == 1u) { s += a[c-W]; n += 1.0; }
      if (k < prm.nz-1u && av[c+W] == 1u) { s += a[c+W]; n += 1.0; }
      if (n > 0.0) { o[c] = s/n; ov[c] = 1u; } else { o[c] = 0.0; ov[c] = 0u; }`),
    extrapW: K([['a', 'f'], ['av', 'u'], ['o', 'f'], ['ov', 'u']], `let N = prm.nx; if (c >= N*(prm.nz+1u)) {return;} let i = c % N; let k = c / N;
      if (av[c] == 1u) { o[c] = a[c]; ov[c] = 1u; return; }
      var s = 0.0; var n = 0.0;
      if (i > 0u && av[c-1u] == 1u) { s += a[c-1u]; n += 1.0; }
      if (i < N-1u && av[c+1u] == 1u) { s += a[c+1u]; n += 1.0; }
      if (k > 0u && av[c-N] == 1u) { s += a[c-N]; n += 1.0; }
      if (k < prm.nz && av[c+N] == 1u) { s += a[c+N]; n += 1.0; }
      if (n > 0.0) { o[c] = s/n; ov[c] = 1u; } else { o[c] = 0.0; ov[c] = 0u; }`),
    advectU: K([['u', 'f'], ['w', 'f'], ['o', 'f']], `let W = prm.nx+1u; let N = prm.nx; if (c >= W*prm.nz) {return;} let i = c % W; let k = c / W;
      if (i == 0u || i == prm.nx) { o[c] = u[c]; return; }
      let x = f32(i)*prm.dx; let z = (f32(k)+0.5)*prm.dx; let dt = prm.dt;
      let uu = u[c]; let ww = 0.25*(w[i-1u+N*k] + w[i+N*k] + w[i-1u+N*(k+1u)] + w[i+N*(k+1u)]);
      let xm = x - 0.5*dt*uu; let zm = z - 0.5*dt*ww;
      o[c] = sU(x - dt*sU(xm, zm), z - dt*sW(xm, zm));`, SAMPLE),
    advectW: K([['u', 'f'], ['w', 'f'], ['o', 'f']], `let W = prm.nx+1u; let N = prm.nx; if (c >= N*(prm.nz+1u)) {return;} let i = c % N; let k = c / N;
      if (k == 0u || k == prm.nz) { o[c] = 0.0; return; }
      let x = (f32(i)+0.5)*prm.dx; let z = f32(k)*prm.dx; let dt = prm.dt;
      let ww = w[c]; let uu = 0.25*(u[i+W*(k-1u)] + u[i+1u+W*(k-1u)] + u[i+W*k] + u[i+1u+W*k]);
      let xm = x - 0.5*dt*uu; let zm = z - 0.5*dt*ww;
      o[c] = sW(x - dt*sU(xm, zm), z - dt*sW(xm, zm));`, SAMPLE),
    // Conservative momentum advection (finite volume on the staggered grid): fluxes through each
    // velocity cell's faces, second-order upwind with a minmod limiter, forward Euler (TVD at
    // CFL <= 0.5). It conserves momentum, which is what sets the depths across a hydraulic jump.
    advFU: K([['u', 'f'], ['w', 'f'], ['o', 'f'], ['typ', 'u']], `let W = prm.nx+1u; let N = prm.nx; if (c >= W*prm.nz) {return;} let i = c % W; let k = c / W;
      if (i == 0u || i == prm.nx) { o[c] = u[c]; return; }
      // fall back to semi-Lagrangian next to the free surface (any air cell in the stencil) and
      // wherever the local Courant number is above 0.45 (the flux form is only stable below 0.5)
      var nearAir = false;
      for (var dk = -1; dk <= 1; dk++) { for (var di = -2; di <= 1; di++) { if (isAir(i32(i) + di, i32(k) + dk)) { nearAir = true; } } }
      let cmax = max(max(abs(u[c-1u]), abs(u[c])), max(abs(u[c+1u]), max(abs(w[i-1u+N*k]), abs(w[i+N*(k+1u)])))) * prm.dt / prm.dx;
      if (nearAir || cmax > 0.45) {
        let x = f32(i)*prm.dx; let z = (f32(k)+0.5)*prm.dx; let dt = prm.dt;
        let ww = 0.25*(w[i-1u+N*k] + w[i+N*k] + w[i-1u+N*(k+1u)] + w[i+N*(k+1u)]);
        let xm = x - 0.5*dt*u[c]; let zm = z - 0.5*dt*ww;
        o[c] = sU(x - dt*sU(xm, zm), z - dt*sW(xm, zm)); return;
      }
      let uc = u[c];
      // east and west faces of the u-cell are cell centres i and i-1
      let ue = 0.5*(uc + u[c+1u]); let uw = 0.5*(u[c-1u] + uc);
      var qe: f32; if (ue > 0.0) { var d2 = 0.0; d2 = u[c+1u] - uc; qe = uc + 0.5*mm(uc - u[c-1u], d2); }
                   else { var d3 = 0.0; if (i + 2u <= prm.nx) { d3 = u[c+2u] - u[c+1u]; } qe = u[c+1u] - 0.5*mm(u[c+1u] - uc, d3); }
      var qw: f32; if (uw > 0.0) { var d1 = 0.0; if (i >= 2u) { d1 = u[c-1u] - u[c-2u]; } qw = u[c-1u] + 0.5*mm(d1, uc - u[c-1u]); }
                   else { qw = uc - 0.5*mm(uc - u[c-1u], u[c+1u] - uc); }
      // north and south faces are corners: w averaged over the two cells either side
      var fluxN = 0.0; var fluxS = 0.0;
      if (k + 1u < prm.nz) {
        let vn = 0.5*(w[i-1u+N*(k+1u)] + w[i+N*(k+1u)]);
        var q: f32; if (vn > 0.0) { var dm = 0.0; if (k > 0u) { dm = uc - u[c-W]; } q = uc + 0.5*mm(dm, u[c+W] - uc); }
                    else { var dp = 0.0; if (k + 2u < prm.nz) { dp = u[c+2u*W] - u[c+W]; } q = u[c+W] - 0.5*mm(u[c+W] - uc, dp); }
        fluxN = vn*q; }
      if (k > 0u) {
        let vs = 0.5*(w[i-1u+N*k] + w[i+N*k]);
        var q: f32; if (vs > 0.0) { var dm = 0.0; if (k >= 2u) { dm = u[c-W] - u[c-2u*W]; } q = u[c-W] + 0.5*mm(dm, uc - u[c-W]); }
                    else { var dp = 0.0; if (k + 1u < prm.nz) { dp = u[c+W] - uc; } q = uc - 0.5*mm(uc - u[c-W], dp); }
        fluxS = vs*q; }
      o[c] = uc - prm.dt/prm.dx*(ue*qe - uw*qw + fluxN - fluxS);`, MM + SAMPLE + AIR),
    advFW: K([['u', 'f'], ['w', 'f'], ['o', 'f'], ['typ', 'u']], `let W = prm.nx+1u; let N = prm.nx; if (c >= N*(prm.nz+1u)) {return;} let i = c % N; let k = c / N;
      if (k == 0u || k == prm.nz) { o[c] = 0.0; return; }
      var nearAir = false;
      for (var dk = -2; dk <= 1; dk++) { for (var di = -1; di <= 1; di++) { if (isAir(i32(i) + di, i32(k) + dk)) { nearAir = true; } } }
      let cmax = max(max(abs(w[c-N]), abs(w[c])), max(abs(w[c+N]), max(abs(u[i+W*k]), abs(u[i+1u+W*(k-1u)])))) * prm.dt / prm.dx;
      if (nearAir || cmax > 0.45) {
        let x = (f32(i)+0.5)*prm.dx; let z = f32(k)*prm.dx; let dt = prm.dt;
        let uu = 0.25*(u[i+W*(k-1u)] + u[i+1u+W*(k-1u)] + u[i+W*k] + u[i+1u+W*k]);
        let xm = x - 0.5*dt*uu; let zm = z - 0.5*dt*w[c];
        o[c] = sW(x - dt*sU(xm, zm), z - dt*sW(xm, zm)); return;
      }
      let wc = w[c];
      // north and south faces of the w-cell are cell centres k and k-1
      let wn = 0.5*(wc + w[c+N]); let ws = 0.5*(w[c-N] + wc);
      var qn: f32; if (wn > 0.0) { qn = wc + 0.5*mm(wc - w[c-N], w[c+N] - wc); }
                   else { var d3 = 0.0; if (k + 2u <= prm.nz) { d3 = w[c+2u*N] - w[c+N]; } qn = w[c+N] - 0.5*mm(w[c+N] - wc, d3); }
      var qs: f32; if (ws > 0.0) { var d1 = 0.0; if (k >= 2u) { d1 = w[c-N] - w[c-2u*N]; } qs = w[c-N] + 0.5*mm(d1, wc - w[c-N]); }
                   else { qs = wc - 0.5*mm(wc - w[c-N], w[c+N] - wc); }
      // east and west faces are corners: u averaged over the two cells above and below
      var fluxE = 0.0; var fluxW = 0.0;
      if (i + 1u < N) {
        let ve = 0.5*(u[i+1u+W*(k-1u)] + u[i+1u+W*k]);
        var q: f32; if (ve > 0.0) { var dm = 0.0; if (i > 0u) { dm = wc - w[c-1u]; } q = wc + 0.5*mm(dm, w[c+1u] - wc); }
                    else { var dp = 0.0; if (i + 2u < N) { dp = w[c+2u] - w[c+1u]; } q = w[c+1u] - 0.5*mm(w[c+1u] - wc, dp); }
        fluxE = ve*q; }
      if (i > 0u) {
        let vw = 0.5*(u[i+W*(k-1u)] + u[i+W*k]);
        var q: f32; if (vw > 0.0) { var dm = 0.0; if (i >= 2u) { dm = w[c-1u] - w[c-2u]; } q = w[c-1u] + 0.5*mm(dm, wc - w[c-1u]); }
                    else { var dp = 0.0; if (i + 1u < N) { dp = w[c+1u] - wc; } q = wc - 0.5*mm(wc - w[c-1u], dp); }
        fluxW = vw*q; }
      o[c] = wc - prm.dt/prm.dx*(wn*qn - ws*qs + fluxE - fluxW);`, MM + SAMPLE + AIR),
    // MacCormack correction for the faces that took the semi-Lagrangian path (next to the surface
    // or at a high local Courant number): trace forward from the new value, correct by half the
    // round-trip error, and clamp to the four values around the departure point.
    // Reads u (old), u2 (after advection); writes o.
    macU: K([['u', 'f'], ['w', 'f'], ['u2', 'f'], ['o', 'f'], ['typ', 'u']], `let W = prm.nx+1u; let N = prm.nx; if (c >= W*prm.nz) {return;} let i = c % W; let k = c / W;
      if (i == 0u || i == prm.nx) { o[c] = u2[c]; return; }
      var nearAir = false;
      for (var dk = -1; dk <= 1; dk++) { for (var di = -2; di <= 1; di++) { if (isAir(i32(i) + di, i32(k) + dk)) { nearAir = true; } } }
      let cmax = max(max(abs(u[c-1u]), abs(u[c])), max(abs(u[c+1u]), max(abs(w[i-1u+N*k]), abs(w[i+N*(k+1u)])))) * prm.dt / prm.dx;
      if (!(nearAir || cmax > 0.45)) { o[c] = u2[c]; return; }
      let x = f32(i)*prm.dx; let z = (f32(k)+0.5)*prm.dx; let dt = prm.dt;
      let ww = 0.25*(w[i-1u+N*k] + w[i+N*k] + w[i-1u+N*(k+1u)] + w[i+N*(k+1u)]);
      let xm = x - 0.5*dt*u[c]; let zm = z - 0.5*dt*ww;
      let vx = sU(xm, zm); let vz = sW(xm, zm);
      let xd = x - dt*vx; let zd = z - dt*vz;
      // forward trace of the advected field from the departure point back to here
      let back = sU2(xd + dt*vx, zd + dt*vz);
      let corr = u2[c] + 0.5*(u[c] - back);
      // limiter: stay within the old values around the departure point
      let fx = clamp(xd/prm.dx, 0.0, f32(prm.nx)); let fz = clamp(zd/prm.dx - 0.5, 0.0, f32(prm.nz) - 1.0);
      let i0 = min(u32(fx), prm.nx - 1u); let k0 = min(u32(fz), max(prm.nz, 2u) - 2u); let k1 = min(k0 + 1u, prm.nz - 1u);
      let a = u[i0+W*k0]; let b = u[i0+1u+W*k0]; let cc = u[i0+W*k1]; let d = u[i0+1u+W*k1];
      o[c] = clamp(corr, min(min(a, b), min(cc, d)), max(max(a, b), max(cc, d)));`, SAMPLE + SAMPLE2U + AIR),
    macW: K([['u', 'f'], ['w', 'f'], ['w2', 'f'], ['o', 'f'], ['typ', 'u']], `let W = prm.nx+1u; let N = prm.nx; if (c >= N*(prm.nz+1u)) {return;} let i = c % N; let k = c / N;
      if (k == 0u || k == prm.nz) { o[c] = w2[c]; return; }
      var nearAir = false;
      for (var dk = -2; dk <= 1; dk++) { for (var di = -1; di <= 1; di++) { if (isAir(i32(i) + di, i32(k) + dk)) { nearAir = true; } } }
      let cmax = max(max(abs(w[c-N]), abs(w[c])), max(abs(w[c+N]), max(abs(u[i+W*k]), abs(u[i+1u+W*(k-1u)])))) * prm.dt / prm.dx;
      if (!(nearAir || cmax > 0.45)) { o[c] = w2[c]; return; }
      let x = (f32(i)+0.5)*prm.dx; let z = f32(k)*prm.dx; let dt = prm.dt;
      let uu = 0.25*(u[i+W*(k-1u)] + u[i+1u+W*(k-1u)] + u[i+W*k] + u[i+1u+W*k]);
      let xm = x - 0.5*dt*uu; let zm = z - 0.5*dt*w[c];
      let vx = sU(xm, zm); let vz = sW(xm, zm);
      let xd = x - dt*vx; let zd = z - dt*vz;
      let back = sW2(xd + dt*vx, zd + dt*vz);
      let corr = w2[c] + 0.5*(w[c] - back);
      let fx = clamp(xd/prm.dx - 0.5, 0.0, f32(prm.nx) - 1.0); let fz = clamp(zd/prm.dx, 0.0, f32(prm.nz));
      let i0 = min(u32(fx), prm.nx - 2u); let k0 = min(u32(fz), prm.nz - 1u); let i1 = min(i0 + 1u, prm.nx - 1u);
      let a = w[i0+N*k0]; let b = w[i1+N*k0]; let cc = w[i0+N*(k0+1u)]; let d = w[i1+N*(k0+1u)];
      o[c] = clamp(corr, min(min(a, b), min(cc, d)), max(max(a, b), max(cc, d)));`, SAMPLE + SAMPLE2W + AIR),
    // ── sediment: the bed is a height zs per column over the erodible zone (zfix .. zcap) ──
    // sp: 0-2 grain d (m), 3-5 fall velocity (m/s), 6-8 critical Shields, 9 bed speed-up,
    //     10 porosity, 11 tan(angle of repose), 12 slope factor, 13 on, 14 feed, 15 s-1
    // per column: bed shear from the log law at the first water cell above the bed, and
    // Meyer-Peter & Mueller bedload per class, weighted by the mix of the top sediment cell
    sedShear: K([['u', 'f'], ['f', 'f'], ['solid', 'u'], ['bed', 'f'], ['cell', 'f'], ['qc', 'f'], ['ust', 'f'], ['SP', 'U']], `let i = c; if (i >= prm.nx) {return;}
      let N = prm.nx; let W = prm.nx + 1u; let dx = prm.dx; let z = bed[3u*(i)];
      qc[3u*i] = 0.0; qc[3u*i+1u] = 0.0; qc[3u*i+2u] = 0.0; ust[i] = 0.0;
      if (z < 0.0 || spv(13u) < 0.5) { return; }
      let k1 = u32(clamp(ceil(z/dx - 0.5), 0.0, f32(prm.nz) - 1.0));
      let c1 = i + N*k1; if (solid[c1] != 0u || f[c1] < 0.5) { return; }
      // reference velocity: the first water cell, or the one above it when the first is less than half
      // a cell above the bed (a cell that has just opened has hardly any velocity yet, and reading it
      // stalls erosion at every cell boundary)
      var kr = k1; var y = (f32(k1) + 0.5)*dx - z;
      if (y < 0.5*dx && k1 + 1u < prm.nz && solid[c1+N] == 0u && f[c1+N] > 0.5) { kr = k1 + 1u; y = y + dx; }
      y = max(y, 0.3*dx);
      let uc = 0.5*(u[i+W*kr] + u[i+1u+W*kr]);
      var fr = vec3f(0.0); let thick = z - bed[3u*(i)+1u];
      if (k1 > 0u && thick > 1e-6) { let ct = i + N*(k1 - 1u); fr = vec3f(cell[4u*ct], cell[4u*ct+1u], cell[4u*ct+2u]); }
      let tot = fr.x + fr.y + fr.z; if (tot <= 1e-6) { return; }
      fr = fr/tot;
      let dm = fr.x*spv(0u) + fr.y*spv(1u) + fr.z*spv(2u);
      let ustar = 0.41*abs(uc)/max(log(30.0*y/max(2.5*dm, 1e-5)), 2.0);
      ust[i] = ustar;
      let R = spv(15u); let g = prm.g;
      for (var k = 0u; k < 3u; k++) {
        let d = spv(k); let th = ustar*ustar/(R*g*d); let ex = max(th - spv(6u+k), 0.0);
        qc[3u*i+k] = fr[k]*8.0*pow(ex, 1.5)*sqrt(R*g*d*d*d)*sign(uc);
      }`),
    // per face between columns: upwind bedload, slope correction, never more than the donor has
    sedFlux: K([['qc', 'f'], ['bed', 'f'], ['q', 'f'], ['SP', 'U']], `let j = c; if (j > prm.nx) {return;}
      let dtm = max(spv(9u)*prm.dt, 1e-9); let dx = prm.dx; let por = spv(10u);
      for (var k = 0u; k < 3u; k++) {
        var qf = 0.0;
        if (j == 0u) { if (spv(14u) > 0.5 && bed[3u*(0)] >= 0.0) { qf = max(qc[k], 0.0); } }
        else if (j == prm.nx) { if (bed[3u*(j-1u)] >= 0.0 && prm.mode < 0.5) { qf = max(qc[3u*(j-1u)+k], 0.0); } }   // a wave flume's far end is closed
        else {
          let za = bed[3u*(j-1u)]; let zb = bed[3u*(j)];
          if (za >= 0.0 && zb >= 0.0) {
            let a = qc[3u*(j-1u)+k]; let b = qc[3u*j+k];
            qf = select(b, a, a + b > 0.0);
            qf = qf*max(1.0 - spv(12u)*sign(qf)*(zb - za)/dx, 0.0);
            let dn = select(j, j - 1u, qf > 0.0);
            let avail = max(bed[3u*(dn)] - bed[3u*(dn)+1u], 0.0)*(1.0 - por)*dx/dtm;
            qf = clamp(qf, -0.5*avail, 0.5*avail);
          }
        }
        q[3u*j+k] = qf;
      }`),
    // per column: Exner (bedload divergence plus net deposition from suspension), and the mix of
    // what is arriving, for new deposits
    sedExner: K([['q', 'f'], ['bed', 'f'], ['colx', 'f'], ['cell', 'f'], ['SP', 'U']], `let i = c; if (i >= prm.nx) {return;}
      if (bed[3u*(i)] < 0.0 || spv(13u) < 0.5) { return; }
      let dtm = spv(9u)*prm.dt; let dx = prm.dx;
      var dV = 0.0; var inc = vec3f(0.0);
      for (var k = 0u; k < 3u; k++) {
        let qa = q[3u*i+k]; let qb = q[3u*(i+1u)+k]; let s2 = colx[6u*i+k];
        dV += -(qb - qa)/dx + s2;
        inc[k] = max(qa, 0.0) + max(-qb, 0.0) + max(s2*dx, 0.0);
      }
      let dz = clamp(dtm*dV/(1.0 - spv(10u)), -0.2*dx, 0.2*dx);
      let z0 = bed[3u*(i)];
      // active layer (Hirano): the top sediment cell's mix changes class by class with what leaves
      // and arrives, so selective transport sorts the bed (fines leave, gravel stays: armouring)
      if (z0 - bed[3u*(i)+1u] > 1e-6) {
        // the top sediment cell: the one under the first water cell (the lowest cell whose centre is at
        // or above the bed surface; cells below are sediment, as sedCells flips them), as in sedShear
        let k1 = u32(clamp(ceil(z0/dx - 0.5), 1.0, f32(prm.nz) - 1.0)); let ct = i + prm.nx*(k1 - 1u);
        var m = vec3f(cell[4u*ct], cell[4u*ct+1u], cell[4u*ct+2u]); let La = dx;
        for (var k = 0u; k < 3u; k++) {
          let dzk = dtm*(-(q[3u*(i+1u)+k] - q[3u*i+k])/dx + colx[6u*i+k])/(1.0 - spv(10u));
          m[k] = max(m[k]*La + dzk, 0.0);
        }
        let mt = m.x + m.y + m.z;
        if (mt > 1e-9) { cell[4u*ct] = m.x/mt; cell[4u*ct+1u] = m.y/mt; cell[4u*ct+2u] = m.z/mt; }
      }
      bed[3u*(i)] = clamp(z0 + dz, bed[3u*(i)+1u], max(bed[3u*(i)+2u] - 0.01*dx, bed[3u*(i)+1u]));
      let t = inc.x + inc.y + inc.z;
      if (t > 1e-12) { colx[6u*i+3u] = inc.x/t; colx[6u*i+4u] = inc.y/t; colx[6u*i+5u] = inc.z/t; }`),
    // slumping: where two columns differ by more than the angle of repose, move half the excess
    // (red-black over faces, so no column is touched twice in one pass)
    sedAval: K([['bed', 'f'], ['SP', 'U']], `let j = 2u*c + PAR + 1u; if (j >= prm.nx) {return;}
      if (spv(13u) < 0.5) { return; }
      let a = j - 1u; let b = j; let za = bed[3u*(a)]; let zb = bed[3u*(b)]; if (za < 0.0 || zb < 0.0) { return; }
      let d = za - zb; let mx = spv(11u)*prm.dx; if (abs(d) <= mx) { return; }
      var mv = 0.5*(abs(d) - mx);
      if (d > 0.0) { mv = min(mv, min(za - bed[3u*(a)+1u], bed[3u*(b)+2u] - 0.01*prm.dx - zb)); mv = max(mv, 0.0); bed[3u*(a)] = za - mv; bed[3u*(b)] = zb + mv; }
      else { mv = min(mv, min(zb - bed[3u*(b)+1u], bed[3u*(a)+2u] - 0.01*prm.dx - za)); mv = max(mv, 0.0); bed[3u*(b)] = zb - mv; bed[3u*(a)] = za + mv; }`, 'override PAR: u32;\n'),
    // cells follow the bed: water <-> sediment inside the erodible zone; new sediment records its
    // mix, the time it was laid down, and its roughness; water it displaces moves up a cell
    sedCells: K([['solid', 'u'], ['f', 'f'], ['bed', 'f'], ['cell', 'f'], ['colx', 'f'], ['tb', 'f'], ['ks', 'f'], ['cs', 'f'], ['SP', 'U']], `if (c >= prm.nx*prm.nz) {return;}
      if (spv(13u) < 0.5) { return; }
      let N = prm.nx; let i = c % N; let k = c / N; let z = bed[3u*(i)]; if (z < 0.0) { return; }
      let zc = (f32(k) + 0.5)*prm.dx;
      if (zc <= bed[3u*(i)+1u] || zc >= bed[3u*(i)+2u]) { return; }
      let want = zc < z; let isS = solid[c] != 0u;
      if (want && !isS) {
        solid[c] = 1u;
        if (k + 1u < prm.nz) { f[c+N] = min(1.0, f[c+N] + f[c]); }
        f[c] = 0.0;
        // a new cell continues the active layer under it (which already took in what arrived);
        // only on a bare floor does it take the arriving mix
        var m = vec3f(colx[6u*i+3u], colx[6u*i+4u], colx[6u*i+5u]);
        if (k > 0u && solid[c-N] != 0u) {
          let mb = vec3f(cell[4u*(c-N)], cell[4u*(c-N)+1u], cell[4u*(c-N)+2u]); let tb2 = mb.x + mb.y + mb.z;
          if (tb2 > 1e-6) { m = mb/tb2; }
        }
        cell[4u*c] = m.x; cell[4u*c+1u] = m.y; cell[4u*c+2u] = m.z; cell[4u*c+3u] = tb[0]*spv(9u);
        ks[c] = 2.5*(m.x*spv(0u) + m.y*spv(1u) + m.z*spv(2u));
        cs[3u*c] = 0.0; cs[3u*c+1u] = 0.0; cs[3u*c+2u] = 0.0;
      } else if (!want && isS) {
        // the half cell of sediment left under the surface merges into the new top cell below
        if (k > 0u && solid[c-N] != 0u) {
          let ma = vec3f(cell[4u*c], cell[4u*c+1u], cell[4u*c+2u]); let mb = vec3f(cell[4u*(c-N)], cell[4u*(c-N)+1u], cell[4u*(c-N)+2u]);
          let ta = ma.x + ma.y + ma.z; let tbb = mb.x + mb.y + mb.z;
          if (ta > 1e-6 && tbb > 1e-6) { let mm2 = (ma/ta + 2.0*mb/tbb)/3.0; cell[4u*(c-N)] = mm2.x; cell[4u*(c-N)+1u] = mm2.y; cell[4u*(c-N)+2u] = mm2.z; }
        }
        solid[c] = 0u;
        f[c] = select(0.0, 1.0, k + 1u < prm.nz && f[min(c + N, prm.nx*prm.nz - 1u)] > 0.5);
        cs[3u*c] = 0.0; cs[3u*c+1u] = 0.0; cs[3u*c+2u] = 0.0;
      }`),
    // suspended load per class: carried by the flow, settling at the fall velocity, mixed by
    // K = kappa u* z' (1 - z'/h) + nu (z' above the bed, h the local depth: the classic
    // parabolic profile behind the Rouse equation). First-order upwind, conservative; nothing
    // crosses into air, solids or the bed (the bed exchange is susBed). Clear water comes in.
    susAdv: K([['u', 'f'], ['w', 'f'], ['typ', 'u'], ['cs', 'f'], ['co', 'f'], ['ust', 'f'], ['bed', 'f'], ['eta', 'f'], ['SP', 'U']], `if (c >= prm.nx*prm.nz) {return;}
      let N = prm.nx; let W = prm.nx + 1u; let i = c % N; let k = c / N; let dx = prm.dx;
      if (typ[c] != 1u || spv(13u) < 0.5) { co[3u*c] = cs[3u*c]; co[3u*c+1u] = cs[3u*c+1u]; co[3u*c+2u] = cs[3u*c+2u]; return; }
      let zb = max(bed[3u*i], 0.0); let h = max(eta[i] - zb, dx); let us = ust[i];
      for (var q = 0u; q < 3u; q++) {
        let ws = spv(3u + q); let C = cs[3u*c+q];
        // second-order upwind with a minmod limiter (first order next to walls, air and the bed)
        var fe = 0.0;
        if (i + 1u < N) { let cn = c + 1u; if (typ[cn] == 1u) { let ue = u[i+1u+W*k];
          let ce = select(face(cv(i32(i)+2, i32(k), q, cs[3u*cn+q]), cs[3u*cn+q], C), face(cv(i32(i)-1, i32(k), q, C), C, cs[3u*cn+q]), ue > 0.0);
          fe = ue*ce - kmix(k, us, zb, h)*(cs[3u*cn+q] - C)/dx; } }
        else if (prm.mode < 0.5) { fe = max(u[i+1u+W*k], 0.0)*C; }
        var fw = 0.0;
        if (i > 0u) { let cn = c - 1u; if (typ[cn] == 1u) { let uw = u[i+W*k];
          let cw = select(face(cv(i32(i)+1, i32(k), q, C), C, cs[3u*cn+q]), face(cv(i32(i)-2, i32(k), q, cs[3u*cn+q]), cs[3u*cn+q], C), uw > 0.0);
          fw = uw*cw - kmix(k, us, zb, h)*(C - cs[3u*cn+q])/dx; } }
        else { fw = min(u[i+W*k], 0.0)*C; }             // clear water in; outflow at the inlet carries C
        var fN = 0.0;
        if (k + 1u < prm.nz) { let cn = c + N; if (typ[cn] == 1u) { let wn = w[i+N*(k+1u)] - ws;
          let cnf = select(face(cv(i32(i), i32(k)+2, q, cs[3u*cn+q]), cs[3u*cn+q], C), face(cv(i32(i), i32(k)-1, q, C), C, cs[3u*cn+q]), wn > 0.0);
          fN = wn*cnf - kmixz(k + 1u, us, zb, h)*(cs[3u*cn+q] - C)/dx; } }
        var fs = 0.0;
        if (k > 0u) { let cn = c - N; if (typ[cn] == 1u) { let wsf = w[i+N*k] - ws;
          let csf = select(face(cv(i32(i), i32(k)+1, q, C), C, cs[3u*cn+q]), face(cv(i32(i), i32(k)-2, q, cs[3u*cn+q]), cs[3u*cn+q], C), wsf > 0.0);
          fs = wsf*csf - kmixz(k, us, zb, h)*(C - cs[3u*cn+q])/dx; } }
        co[3u*c+q] = max(C - prm.dt/dx*(fe - fw + fN - fs), 0.0);
      }`, MM + `fn kmix(k:u32, us:f32, zb:f32, h:f32) -> f32 { let z = clamp((f32(k) + 0.5)*prm.dx - zb, 0.0, h); return 0.41*us*z*(1.0 - z/h) + 1e-6; }
fn kmixz(k:u32, us:f32, zb:f32, h:f32) -> f32 { let z = clamp(f32(k)*prm.dx - zb, 0.0, h); return 0.41*us*z*(1.0 - z/h) + 1e-6; }
// the concentration in cell (i, k), or the fallback where that cell is not water (zero gradient)
fn cv(i:i32, k:i32, q:u32, fb:f32) -> f32 { if (i < 0 || k < 0 || i >= i32(prm.nx) || k >= i32(prm.nz)) { return fb; } let c = u32(i) + prm.nx*u32(k); if (typ[c] != 1u) { return fb; } return cs[3u*c+q]; }
// limited face value from the cell upstream of the donor, the donor, and the receiver
// (van Leer limiter: minmod, which always takes the smaller slope, flattened the Rouse profile
// by about 18% in its exponent; van Leer's harmonic mean is less biased on smooth profiles)
fn vl(a:f32, b:f32) -> f32 { if (a*b <= 0.0) { return 0.0; } return 2.0*a*b/(a + b); }
fn face(cuu:f32, cu:f32, cd:f32) -> f32 { return cu + 0.5*vl(cu - cuu, cd - cu); }
`),
    // bed exchange per column: pickup E = ws c_a (van Rijn 1984 reference concentration, only
    // where u* > ws), settling D = ws C at the first water cell; the net (D - E) goes to Exner
    susBed: K([['f', 'f'], ['solid', 'u'], ['bed', 'f'], ['cell', 'f'], ['cs', 'f'], ['colx', 'f'], ['ust', 'f'], ['ustK', 'f'], ['SP', 'U']], `let i = c; if (i >= prm.nx) {return;}
      colx[6u*i] = 0.0; colx[6u*i+1u] = 0.0; colx[6u*i+2u] = 0.0;
      let z = bed[3u*i]; if (z < 0.0 || spv(13u) < 0.5) { ustK[i] = 0.0; return; }
      let N = prm.nx; let dx = prm.dx;
      let k1 = u32(clamp(ceil(z/dx - 0.5), 0.0, f32(prm.nz) - 1.0)); let c1 = i + N*k1;
      if (solid[c1] != 0u || f[c1] < 0.5) { ustK[i] = 0.0; return; }
      // mixing in the water column follows a smoothed u*, relaxed over one eddy turnover (h / u*,
      // kept between 1 and 30 s): depth-scale mixing does not follow every flicker of the bed shear,
      // and a profile that switches with it averages flatter than Rouse
      var kt = k1; while (kt + 1u < prm.nz && solid[i + N*(kt + 1u)] == 0u && f[i + N*(kt + 1u)] > 0.5) { kt++; }
      let hl = max(f32(kt + 1u)*dx - z, dx);
      let tau = clamp(hl/max(ustK[i], 1e-3), 1.0, 30.0);
      ustK[i] = ustK[i] + (ust[i] - ustK[i])*min(prm.dt/tau, 1.0);
      var fr = vec3f(0.0);
      if (k1 > 0u && z - bed[3u*i+1u] > 1e-6) { let ct = c1 - N; fr = vec3f(cell[4u*ct], cell[4u*ct+1u], cell[4u*ct+2u]); }
      let tot = fr.x + fr.y + fr.z; if (tot > 1e-6) { fr = fr/tot; }
      let us = ust[i]; let a = max((f32(k1) + 0.5)*dx - z, 0.3*dx); let R = spv(15u);
      for (var q = 0u; q < 3u; q++) {
        let d = spv(q); let ws = spv(3u + q); let thc = spv(6u + q);
        var E = 0.0;
        if (us > ws && tot > 1e-6) {
          let th = us*us/(R*prm.g*d); let T = max(th/thc - 1.0, 0.0);
          let Ds = d*pow(R*prm.g/1e-12, 1.0/3.0);
          let ca = min(0.015*d*pow(T, 1.5)/(a*pow(Ds, 0.3)), 0.05);
          E = fr[q]*ws*ca;
        }
        let Cb = cs[3u*c1+q]; var D = ws*Cb;
        // a cell cannot drop more than it holds, or pick up more than the bed can give
        D = min(D, Cb*dx/prm.dt);
        cs[3u*c1+q] = max(Cb + prm.dt*(E - D)/dx, 0.0);
        colx[6u*i+q] = D - E;
      }`),
    // flexible plants (Luhar & Nepf 2011), per column: the Cauchy number Ca = rho Cd d u^2 l^3 / EI
    // from the mean velocity over the plants gives the effective length
    // l_e / l = 1 - (1 - 0.9 Ca^(-1/3)) / (1 + 8 Ca^(-3/2)); bent plants drag only over their lower
    // l_e (pf = 1 below base + l_e, 0 above). Quasi-static: under waves the meadow sways with each wave.
    // pl per column: base (m), length (m), stem d (m), Cd, EI (N m2); out: l_e/l and flow direction
    plantBend: K([['u', 'f'], ['pl', 'f'], ['pf', 'f'], ['po', 'f']], `let i = c; if (i >= prm.nx) {return;}
      let N = prm.nx; let W = prm.nx + 1u; let dx = prm.dx;
      let base = pl[5u*i]; let l = pl[5u*i+1u];
      if (l <= 0.0) { po[2u*i] = 1.0; po[2u*i+1u] = 0.0; return; }
      let k0 = u32(floor(base/dx)); let k1 = min(u32(ceil((base + l)/dx)), prm.nz);
      var us = 0.0; var n = 0.0;
      for (var k = k0; k < k1; k++) { us += 0.5*(u[i+W*k] + u[i+1u+W*k]); n += 1.0; }
      let U = select(0.0, us/n, n > 0.0);
      let Ca = 1000.0*pl[5u*i+3u]*pl[5u*i+2u]*U*U*l*l*l/max(pl[5u*i+4u], 1e-12);
      var r = 1.0;
      if (Ca > 1e-6) { r = 1.0 - (1.0 - 0.9*pow(Ca, -1.0/3.0))/(1.0 + 8.0*pow(Ca, -1.5)); }
      r = clamp(r, 0.05, 1.0);
      po[2u*i] = r; po[2u*i+1u] = select(0.0, sign(U), abs(U) > 1e-4);
      let top = base + r*l;
      for (var k = k0; k < k1; k++) { pf[i + N*k] = select(0.0, 1.0, (f32(k) + 0.5)*dx < top); }`),
    // dye: a passive tracer released by the dye tool, carried by the flow (van Leer upwind, the same
    // flux form as the suspended load) with a little diffusion; nothing crosses into air or solids
    dyeAdv: K([['u', 'f'], ['w', 'f'], ['typ', 'u'], ['cs', 'f'], ['co', 'f']], `if (c >= prm.nx*prm.nz) {return;}
      let N = prm.nx; let W = prm.nx + 1u; let i = c % N; let k = c / N; let dx = prm.dx; let q = 0u;
      if (typ[c] != 1u) { co[c] = cs[c]*select(1.0, 0.0, typ[c] == 0u); return; }
      let C = cs[c]; let Kd = 2e-4*dx/0.01;
      var fe = 0.0;
      if (i + 1u < N) { let cn = c + 1u; if (typ[cn] == 1u) { let ue = u[i+1u+W*k];
        let ce = select(face(cv(i32(i)+2, i32(k), q, cs[cn]), cs[cn], C), face(cv(i32(i)-1, i32(k), q, C), C, cs[cn]), ue > 0.0);
        fe = ue*ce - Kd*(cs[cn] - C)/dx; } }
      else { fe = max(u[i+1u+W*k], 0.0)*C; }
      var fw = 0.0;
      if (i > 0u) { let cn = c - 1u; if (typ[cn] == 1u) { let uw = u[i+W*k];
        let cw = select(face(cv(i32(i)+1, i32(k), q, C), C, cs[cn]), face(cv(i32(i)-2, i32(k), q, cs[cn]), cs[cn], C), uw > 0.0);
        fw = uw*cw - Kd*(C - cs[cn])/dx; } }
      else { fw = min(u[i+W*k], 0.0)*C; }
      var fN = 0.0;
      if (k + 1u < prm.nz) { let cn = c + N; if (typ[cn] == 1u) { let wn = w[i+N*(k+1u)];
        let cf = select(face(cv(i32(i), i32(k)+2, q, cs[cn]), cs[cn], C), face(cv(i32(i), i32(k)-1, q, C), C, cs[cn]), wn > 0.0);
        fN = wn*cf - Kd*(cs[cn] - C)/dx; } }
      var fS = 0.0;
      if (k > 0u) { let cn = c - N; if (typ[cn] == 1u) { let wsf = w[i+N*k];
        let cf = select(face(cv(i32(i), i32(k)+1, q, C), C, cs[cn]), face(cv(i32(i), i32(k)-2, q, cs[cn]), cs[cn], C), wsf > 0.0);
        fS = wsf*cf - Kd*(C - cs[cn])/dx; } }
      co[c] = clamp(C - prm.dt/dx*(fe - fw + fN - fS), 0.0, 1.0);`, MM + `fn cv(i:i32, k:i32, q:u32, fb:f32) -> f32 { if (i < 0 || k < 0 || i >= i32(prm.nx) || k >= i32(prm.nz)) { return fb; } let c = u32(i) + prm.nx*u32(k); if (typ[c] != 1u) { return fb; } return cs[c]; }
fn vl(a:f32, b:f32) -> f32 { if (a*b <= 0.0) { return 0.0; } return 2.0*a*b/(a + b); }
fn face(cuu:f32, cu:f32, cd:f32) -> f32 { return cu + 0.5*vl(cu - cuu, cd - cu); }
`),
    // inject dye in a disc (x, z, r, amount) from the DI uniform
    dyeInject: K([['f', 'f'], ['solid', 'u'], ['dye', 'f'], ['DI', 'U']], `if (c >= prm.nx*prm.nz) {return;}
      let N = prm.nx; let x = (f32(c % N) + 0.5)*prm.dx; let z = (f32(c / N) + 0.5)*prm.dx;
      let dx2 = x - spv(0u); let dz2 = z - spv(1u); let r = max(spv(2u), prm.dx);
      let d2 = (dx2*dx2 + dz2*dz2)/(r*r);
      if (d2 > 1.0 || solid[c] != 0u || f[c] < 0.5) { return; }
      dye[c] = max(dye[c], spv(3u)*(1.0 - d2*d2));`),
    // particles, for the look (abstract, like the sediment interface's grains). Each: x, z, -, -, vx,
    // vz, age, kind. Tracers (kind 0) ride the flow in the water and are reborn at random water cells.
    // Foam (kind 1) is born where the water is torn up (strong shear, or water thrown upward at the
    // surface); in the air it flies ballistically, in the water it rides the flow, rises, and fades.
    // PU: 0 dt (s of model time this frame), 1 time, 2 frame, 3 tracers, 4 foam, 5 still depth, 6 -, 7 -
    partStep: K([['u', 'f'], ['w', 'f'], ['f', 'f'], ['solid', 'u'], ['eta', 'f'], ['PT', 'f'], ['PU', 'U']], `let nT = u32(spv(3u)); let nF = u32(spv(4u)); if (c >= nT + nF) {return;}
      let dt = spv(0u); let o = 8u*c; let kind = select(0u, 1u, c >= nT);
      var x = PT[o]; var z = PT[o+1u]; var vx = PT[o+4u]; var vz = PT[o+5u]; var age = PT[o+6u];
      let px0 = x; let pz0 = z; var fresh = false;      // where it was; fresh = born this step (no line drawn)
      var sd = c*747796405u + u32(spv(2u))*2891336453u + 12345u + u32(spv(6u))*2654435761u;
      let life = select(2.5 + 3.5*rnd(&sd), 1.0 + 2.0*rnd(&sd), kind == 1u);
      let L = f32(prm.nx)*prm.dx; let Hd = f32(prm.nz)*prm.dx; let h0 = max(spv(5u), prm.dx);
      if (age >= 0.0) { age += dt; }
      let inside = x > 0.0 && x < L && z > 0.0 && z < Hd;
      let cl = cellOf(x, z); let wet = inside && solid[cl] == 0u && fAt2(x, z) >= 0.5; let dry = inside && solid[cl] == 0u && !wet;
      if (kind == 0u) {
        if (age < 0.0 || age > life || !wet) {
          // reborn at a random water cell (a few tries)
          age = -1.0; fresh = true;
          for (var t = 0u; t < 4u; t++) {
            let xi = rnd(&sd)*L; let zi = rnd(&sd)*Hd; let ci = cellOf(xi, zi);
            if (solid[ci] == 0u && fAt2(xi, zi) >= 0.5) { x = xi; z = zi; age = 0.0; break; }
          }
          let v = vel(x, z); vx = v.x; vz = v.y;
        } else if (dt > 0.0) {
          let v1 = vel(x, z); let v2 = vel(x + 0.5*dt*v1.x, z + 0.5*dt*v1.y);
          x += dt*v2.x; z += dt*v2.y; vx = v2.x; vz = v2.y;
        }
      } else {
        if (age < 0.0 || age > life || !inside || (inside && solid[cl] != 0u)) {
          age = -1.0; fresh = true;
          // try to be born: a random column, at its surface or somewhere in the water below it
          let i = min(u32(rnd(&sd)*f32(prm.nx)), prm.nx - 1u); let e = eta[i];
          if (e > 0.0 && dt > 0.0) {
            let atSurf = rnd(&sd) < 0.6;
            let zb = select(rnd(&sd)*e, e - rnd(&sd)*2.0*prm.dx, atSurf);
            let xb = (f32(i) + rnd(&sd))*prm.dx;
            let cb = cellOf(xb, zb);
            if (solid[cb] == 0u && fAt2(xb, zb) >= 0.5) {
              let v = vel(xb, zb); let sh = shearAt(xb, zb);
              // the local depth (surface down to the first solid) sets the shear and speed scales
              var kd = u32(clamp(e/prm.dx, 0.0, f32(prm.nz) - 1.0)); while (kd > 0u && solid[i + prm.nx*kd] == 0u) { kd--; }
              let hl = max(e - f32(kd + 1u)*prm.dx, 2.0*prm.dx);
              let tsc = sqrt(prm.g/hl);
              // torn-up water: shear well above what the mean flow alone gives (|u|/h), so a smooth fast
              // jet makes none but the eddies of a jump roller or a breaking crest do; only in the upper
              // part of the water, where air gets in
              let near = smoothstep(0.55, 0.8, (zb - f32(kd + 1u)*prm.dx)/hl);
              var m = near*smoothstep(2.5, 5.0, sh/(length(v)/hl + 0.3*tsc));
              let up = smoothstep(0.08, 0.25, v.y/sqrt(prm.g*hl));
              let im = max(i, 1u) - 1u; let ip = min(i + 1u, prm.nx - 1u);
              let slope = select(0.0, abs(eta[ip] - eta[im])/(f32(ip - im)*prm.dx), eta[im] > 0.0 && eta[ip] > 0.0);
              if (atSurf) { m = max(m, max(up, smoothstep(0.25, 0.7, slope))); }
              if (rnd(&sd) < m) {
                x = xb; z = zb; age = 0.0; vx = v.x; vz = v.y;
                if (atSurf && up > 0.0) { vx = v.x*(1.0 + 0.3*rnd(&sd)); vz = v.y*(1.1 + 0.6*rnd(&sd)) + 0.1*sqrt(prm.g*h0)*rnd(&sd); }
              }
            }
          }
        } else if (dt > 0.0) {
          if (dry) { vz -= prm.g*dt; vx *= (1.0 - 0.5*dt); x += vx*dt; z += vz*dt; }
          else if (wet) { let v = vel(x, z); let rise = 0.04*sqrt(prm.g*h0);
            // bubbles dragged a few cells under dissolve quickly, so foam stays a surface feature instead of
            // hanging in curtains or tracing the eddies of a breaker
            let sub = eta[min(u32(max(x/prm.dx, 0.0)), prm.nx - 1u)] - z; age += dt*(4.0*smoothstep(0.15, 0.5, sub/h0) + 8.0*smoothstep(2.0*prm.dx, 6.0*prm.dx, sub));
            vx = v.x; vz = v.y + rise; x += vx*dt; z += vz*dt; if (z > eta[min(u32(max(x/prm.dx, 0.0)), prm.nx - 1u)]) { z = eta[min(u32(max(x/prm.dx, 0.0)), prm.nx - 1u)] - 0.2*prm.dx; } }
        }
      }
      PT[o] = x; PT[o+1u] = z; PT[o+2u] = select(px0, x, fresh); PT[o+3u] = select(pz0, z, fresh); PT[o+4u] = vx; PT[o+5u] = vz; PT[o+6u] = age; PT[o+7u] = f32(kind);`,
      SAMPLE + `fn rnd(s: ptr<function, u32>) -> f32 { *s = *s*747796405u + 2891336453u; var r = ((*s >> ((*s >> 28u) + 4u)) ^ *s)*277803737u; r = (r >> 22u) ^ r; return f32(r)/4294967295.0; }
fn cellOf(x:f32, z:f32) -> u32 { let i = u32(clamp(x/prm.dx, 0.0, f32(prm.nx) - 1.0)); let k = u32(clamp(z/prm.dx, 0.0, f32(prm.nz) - 1.0)); return i + prm.nx*k; }
fn fAt2(x:f32, z:f32) -> f32 { return f[cellOf(x, z)]; }
fn vel(x:f32, z:f32) -> vec2f { return vec2f(sU(x, z), sW(x, z)); }
fn shearAt(x:f32, z:f32) -> f32 { let d = prm.dx; return abs(sU(x, z + d) - sU(x, z - d))/(2.0*d) + abs(sW(x + d, z) - sW(x - d, z))/(2.0*d); }
`),
    // Smoke comb (after Marey's smoke machine): S release points spread over the depth at x = xc, each
    // a ring of M particles; every emission resets slot n of every ring to its release point, so each
    // ring traced in slot order is a streakline. Particles that reach air, a solid or the ends stop.
    // CU: 0 dt, 1 emit now, 2 slot n, 3 xc, 4 S, 5 M, 6 emission count
    combStep: K([['u', 'f'], ['w', 'f'], ['f', 'f'], ['solid', 'u'], ['eta', 'f'], ['CB', 'f'], ['CU', 'U']], `let S = u32(spv(4u)); let Mm = u32(spv(5u)); if (c >= S*Mm) {return;}
      let s = c / Mm; let j = c % Mm; let o = 4u*c; let dt = spv(0u);
      var x = CB[o]; var z = CB[o+1u]; var age = CB[o+2u]; var cnt = CB[o+3u];
      if (spv(1u) > 0.5 && j == u32(spv(2u))) {
        let i = u32(clamp(spv(3u)/prm.dx, 0.0, f32(prm.nx) - 1.0));
        var kb = 0u; while (kb < prm.nz && solid[i + prm.nx*kb] != 0u) { kb++; }
        let zb = f32(kb)*prm.dx; let e = eta[i];
        if (e > zb + 3.0*prm.dx) { x = spv(3u); z = zb + prm.dx + (f32(s) + 0.5)/f32(S)*(e - zb - 2.0*prm.dx); age = 0.0; cnt = spv(6u); } else { age = -1.0; }
      } else if (age >= 0.0 && dt > 0.0) {
        let v1 = vel(x, z); let v2 = vel(x + 0.5*dt*v1.x, z + 0.5*dt*v1.y); x += dt*v2.x; z += dt*v2.y; age += dt;
        let L = f32(prm.nx)*prm.dx; let Hd = f32(prm.nz)*prm.dx;
        if (!(x > 0.0 && x < L && z > 0.0 && z < Hd)) { age = -1.0; } else if (solid[cellOf(x, z)] != 0u || fAt2(x, z) < 0.5) { age = -1.0; }
      }
      CB[o] = x; CB[o+1u] = z; CB[o+2u] = age; CB[o+3u] = cnt;`,
      SAMPLE + `fn cellOf(x:f32, z:f32) -> u32 { let i = u32(clamp(x/prm.dx, 0.0, f32(prm.nx) - 1.0)); let k = u32(clamp(z/prm.dx, 0.0, f32(prm.nz) - 1.0)); return i + prm.nx*k; }
fn fAt2(x:f32, z:f32) -> f32 { return f[cellOf(x, z)]; }
fn vel(x:f32, z:f32) -> vec2f { return vec2f(sU(x, z), sW(x, z)); }
`),
    // Bubble wire (the hydrogen-bubble technique): at each release every wire puts out a row of K
    // marks across the depth at once; the row is a timeline whose bending is the velocity profile.
    // WU: 0 dt, 1 release now, 2 slot n, 3 -, 4 wires, 5 R, 6 K, 7 -, 8..11 wire x
    wireStep: K([['u', 'f'], ['w', 'f'], ['f', 'f'], ['solid', 'u'], ['eta', 'f'], ['WB', 'f'], ['WU', 'U']], `let nW = u32(spv(4u)); let R = u32(spv(5u)); let Kk = u32(spv(6u)); if (c >= nW*R*Kk) {return;}
      let wi = c / (R*Kk); let r = (c / Kk) % R; let k = c % Kk; let o = 4u*c; let dt = spv(0u);
      var x = WB[o]; var z = WB[o+1u]; var age = WB[o+2u];
      if (spv(1u) > 0.5 && r == u32(spv(2u))) {
        let xw = spv(8u + wi); let i = u32(clamp(xw/prm.dx, 0.0, f32(prm.nx) - 1.0));
        var kb = 0u; while (kb < prm.nz && solid[i + prm.nx*kb] != 0u) { kb++; }
        let zb = f32(kb)*prm.dx; let e = eta[i];
        if (e > zb + 3.0*prm.dx) { x = xw; z = zb + 0.5*prm.dx + (f32(k) + 0.5)/f32(Kk)*(e - zb - 1.5*prm.dx); age = 0.0; } else { age = -1.0; }
      } else if (age >= 0.0 && dt > 0.0) {
        let v1 = vel(x, z); let v2 = vel(x + 0.5*dt*v1.x, z + 0.5*dt*v1.y); x += dt*v2.x; z += dt*v2.y; age += dt;
        let L = f32(prm.nx)*prm.dx; let Hd = f32(prm.nz)*prm.dx;
        if (!(x > 0.0 && x < L && z > 0.0 && z < Hd)) { age = -1.0; } else if (solid[cellOf(x, z)] != 0u || fAt2(x, z) < 0.5) { age = -1.0; }
      }
      WB[o] = x; WB[o+1u] = z; WB[o+2u] = age; WB[o+3u] = f32(r);`,
      SAMPLE + `fn cellOf(x:f32, z:f32) -> u32 { let i = u32(clamp(x/prm.dx, 0.0, f32(prm.nx) - 1.0)); let k = u32(clamp(z/prm.dx, 0.0, f32(prm.nz) - 1.0)); return i + prm.nx*k; }
fn fAt2(x:f32, z:f32) -> f32 { return f[cellOf(x, z)]; }
fn vel(x:f32, z:f32) -> vec2f { return vec2f(sU(x, z), sW(x, z)); }
`),
    // river (mode 0): the discharge q comes in over the wet depth at the inlet; the outlet passes
    // q plus a radiating correction toward the tailwater level (hout), so long waves leave.
    // wave (mode 1): piston paddle at x = 0, closed far end (the sponge takes the waves out).
    bcU: K([['u', 'f'], ['solid', 'u'], ['tb', 'f'], ['eta', 'f']], `let W = prm.nx+1u; if (c >= W*prm.nz) {return;} let i = c % W; let k = c / W;
      let z = (f32(k) + 0.5)*prm.dx; let ramp = min(tb[0]*prm.om, 1.0);
      if (i == 0u) {
        if (solid[prm.nx*k] != 0u) { u[c] = 0.0; return; }
        // every face with any part below the inlet surface e carries q / (e - zin); fluxX takes the
        // wet fraction of each face, so exactly q dt comes in per step
        if (prm.mode < 0.5) { let e = max(eta[0], prm.zin + prm.dx); u[c] = select(0.0, prm.q*ramp/(e - prm.zin), f32(k)*prm.dx < e); }
        else {
          // piston paddle; with absorption it also corrects by the level it sees at its face, so
          // waves coming back from the flume are absorbed instead of re-reflected
          let t = tb[0]; let rw = min(t*prm.om/12.566371, 1.0);
          var up = 0.5*prm.S*prm.om*sin(prm.om*t)*rw;
          if (prm.absorb > 0.5 && eta[0] > 0.0) {
            let h = max(prm.hin - prm.zin, prm.dx);
            up -= sqrt(prm.g/h)*((eta[0] - prm.hin) - prm.amp*sin(prm.om*t)*rw);
          }
          u[c] = up;
        }
        return; }
      if (i == prm.nx) {
        if (solid[prm.nx-1u+prm.nx*k] != 0u || prm.mode > 0.5) { u[c] = 0.0; return; }
        let e = max(eta[prm.nx-1u], prm.zout + prm.dx); let dT = max(prm.hout - prm.zout, prm.dx);
        let uo = max(prm.q*ramp/dT + sqrt(prm.g/dT)*(e - prm.hout), 0.0);
        u[c] = select(0.0, uo, z < e); return; }
      if (solid[i-1u+prm.nx*k] != 0u || solid[i+prm.nx*k] != 0u) { u[c] = 0.0; }`),
    bcW: K([['w', 'f'], ['solid', 'u']], `let N = prm.nx; if (c >= N*(prm.nz+1u)) {return;} let i = c % N; let k = c / N;
      if (k == 0u || k == prm.nz) { w[c] = 0.0; return; }
      if (solid[i+N*(k-1u)] != 0u || solid[i+N*k] != 0u) { w[c] = 0.0; }`),
    spongeU: K([['u', 'f']], `let W = prm.nx+1u; if (c >= W*prm.nz) {return;} if (prm.mode < 0.5) {return;}
      let x = f32(c % W)*prm.dx; let L = f32(prm.nx)*prm.dx; if (x <= prm.sx0) {return;}
      let r = (x - prm.sx0)/max(L - prm.sx0, 1e-6); u[c] = u[c]/(1.0 + prm.srate*prm.dt*r*r);`),
    spongeW: K([['w', 'f']], `let N = prm.nx; if (c >= N*(prm.nz+1u)) {return;} if (prm.mode < 0.5) {return;}
      let x = (f32(c % N) + 0.5)*prm.dx; let L = f32(prm.nx)*prm.dx; if (x <= prm.sx0) {return;}
      let r = (x - prm.sx0)/max(L - prm.sx0, 1e-6); w[c] = w[c]/(1.0 + prm.srate*prm.dt*r*r);`),
    rhs: K([['typ', 'u'], ['u', 'f'], ['w', 'f'], ['r', 'f'], ['p', 'f'], ['BU', 'f'], ['BW', 'f']], `if (c >= prm.nx*prm.nz) {return;} let i = c % prm.nx; let k = c / prm.nx; let W = prm.nx+1u; let N = prm.nx;
      p[c] = 0.0;
      if (typ[c] != 1u) { r[c] = 0.0; return; }
      let a = i+W*k; let b = i+N*k;
      let div = (BU[a+1u]*u[a+1u] - BU[a]*u[a] + BW[b+N]*w[b+N] - BW[b]*w[b]) / prm.dx;
      r[c] = -div*prm.dx*prm.dx/prm.dt;`),
    applyA: K([['typ', 'u'], ['X', 'f'], ['q', 'f']], `if (c >= prm.nx*prm.nz) {return;} let i = c % prm.nx; let k = c / prm.nx;
      if (typ[c] != 1u) { q[c] = 0.0; return; } var s = 0.0; let d = nbr(c, i, k, &s); q[c] = d*X[c] - s;`, NB),
    applyA0: K([['typ', 'u'], ['X', 'f'], ['q', 'f'], ['F', 'f'], ['BU', 'f'], ['BW', 'f']], `if (c >= prm.nx*prm.nz) {return;} let i = c % prm.nx; let k = c / prm.nx;
      if (typ[c] != 1u) { q[c] = 0.0; return; } var s = 0.0; let d = nbr(c, i, k, &s); q[c] = d*X[c] - s;`, NBG),
    rbgs0: K([['typ', 'u'], ['X', 'f'], ['b', 'f'], ['F', 'f'], ['BU', 'f'], ['BW', 'f']], `if (c >= prm.nx*prm.nz) {return;} let i = c % prm.nx; let k = c / prm.nx;
      if (typ[c] != 1u || ((i + k) & 1u) != COLOR) { return; }
      var s = 0.0; let d = nbr(c, i, k, &s); if (d > 0.0) { X[c] = (b[c] + s)/d; }`, 'override COLOR: u32;\n' + NBG),
    resid0: K([['typ', 'u'], ['X', 'f'], ['b', 'f'], ['r', 'f'], ['F', 'f'], ['BU', 'f'], ['BW', 'f']], `if (c >= prm.nx*prm.nz) {return;} let i = c % prm.nx; let k = c / prm.nx;
      if (typ[c] != 1u) { r[c] = 0.0; return; } var s = 0.0; let d = nbr(c, i, k, &s); r[c] = b[c] - (d*X[c] - s);`, NBG),
    rbgs: K([['typ', 'u'], ['X', 'f'], ['b', 'f']], `if (c >= prm.nx*prm.nz) {return;} let i = c % prm.nx; let k = c / prm.nx;
      if (typ[c] != 1u || ((i + k) & 1u) != COLOR) { return; }
      var s = 0.0; let d = nbr(c, i, k, &s); if (d > 0.0) { X[c] = (b[c] + s)/d; }`, 'override COLOR: u32;\n' + NB),
    resid: K([['typ', 'u'], ['X', 'f'], ['b', 'f'], ['r', 'f']], `if (c >= prm.nx*prm.nz) {return;} let i = c % prm.nx; let k = c / prm.nx;
      if (typ[c] != 1u) { r[c] = 0.0; return; } var s = 0.0; let d = nbr(c, i, k, &s); r[c] = b[c] - (d*X[c] - s);`, NB),
    coarsen: K([['tf', 'u'], ['tc', 'u']], `if (c >= prm.cnx*prm.cnz) {return;} let i = c % prm.cnx; let k = c / prm.cnx;
      var air = 0u; var liq = 0u;
      for (var dk = 0u; dk < 2u; dk++) { for (var di = 0u; di < 2u; di++) {
        let fi = 2u*i + di; let fk = 2u*k + dk; if (fi >= prm.nx || fk >= prm.nz) { continue; }
        let t = tf[fi + prm.nx*fk]; if (t == 2u) { air++; } else if (t == 1u) { liq++; } } }
      tc[c] = select(select(0u, 1u, liq > 0u), 2u, air > 0u);`),
    restrict: K([['tf', 'u'], ['rf', 'f'], ['tc', 'u'], ['bc', 'f']], `if (c >= prm.cnx*prm.cnz) {return;} let i = c % prm.cnx; let k = c / prm.cnx;
      if (tc[c] != 1u) { bc[c] = 0.0; return; }
      var s = 0.0;
      for (var dk = 0u; dk < 2u; dk++) { for (var di = 0u; di < 2u; di++) {
        let fi = 2u*i + di; let fk = 2u*k + dk; if (fi >= prm.nx || fk >= prm.nz) { continue; }
        let fc = fi + prm.nx*fk; if (tf[fc] == 1u) { s += rf[fc]; } } }
      bc[c] = s;`),
    prolong: K([['tf', 'u'], ['xf', 'f'], ['tc', 'u'], ['xc', 'f']], `if (c >= prm.nx*prm.nz) {return;} let i = c % prm.nx; let k = c / prm.nx;
      if (tf[c] != 1u) { return; } let cc = (i >> 1u) + prm.cnx*(k >> 1u); if (tc[cc] == 1u) { xf[c] += xc[cc]; }`),
    dot: K([['a', 'f'], ['b', 'f'], ['part', 'f']], `let n = prm.nx*prm.nz; var v = 0.0; if (c < n) { v = a[c]*b[c]; } sh[lid.x] = v; workgroupBarrier();
      for (var s = 128u; s > 0u; s >>= 1u) { if (lid.x < s) { sh[lid.x] += sh[lid.x+s]; } workgroupBarrier(); }
      if (lid.x == 0u) { part[wid.x] = sh[0]; }`, 'var<workgroup> sh: array<f32,256>;', 256),
    // OP 0: sum of a, OP 1: max of a
    red: K([['a', 'f'], ['part', 'f']], `let n = prm.nx*prm.nz; var v = 0.0; if (c < n) { v = a[c]; } sh[lid.x] = v; workgroupBarrier();
      for (var s = 128u; s > 0u; s >>= 1u) { if (lid.x < s) { if (OP == 0u) { sh[lid.x] += sh[lid.x+s]; } else { sh[lid.x] = max(sh[lid.x], sh[lid.x+s]); } } workgroupBarrier(); }
      if (lid.x == 0u) { part[wid.x] = sh[0]; }`, 'override OP: u32;\nvar<workgroup> sh: array<f32,256>;', 256),
    fin: K([['part', 'f'], ['scal', 'f']], `let np = (prm.nx*prm.nz + 255u)/256u; var v = 0.0;
      for (var j = lid.x; j < np; j += 256u) { if (OP == 0u) { v += part[j]; } else { v = max(v, part[j]); } } sh[lid.x] = v; workgroupBarrier();
      for (var s = 128u; s > 0u; s >>= 1u) { if (lid.x < s) { if (OP == 0u) { sh[lid.x] += sh[lid.x+s]; } else { sh[lid.x] = max(sh[lid.x], sh[lid.x+s]); } } workgroupBarrier(); }
      if (lid.x == 0u) { scal[SLOT] = sh[0]; }`, 'override SLOT: u32;\noverride OP: u32;\nvar<workgroup> sh: array<f32,256>;', 256),
    speed: K([['u', 'f'], ['w', 'f'], ['typ', 'u'], ['o', 'f']], `if (c >= prm.nx*prm.nz) {return;} let i = c % prm.nx; let k = c / prm.nx; let W = prm.nx+1u;
      if (typ[c] != 1u) { o[c] = 0.0; return; }
      o[c] = max(max(abs(u[i+W*k]), abs(u[i+1u+W*k])), max(abs(w[i+prm.nx*k]), abs(w[i+prm.nx*(k+1u)])));`),
    alpha: K([['scal', 'f']], `if (c == 0u) { scal[2] = select(0.0, scal[0]/scal[1], abs(scal[1]) > 1e-30); }`),
    beta: K([['scal', 'f']], `if (c == 0u) { scal[3] = select(0.0, scal[4]/scal[0], abs(scal[0]) > 1e-30); scal[0] = scal[4]; }`),
    updPR: K([['p', 'f'], ['r', 'f'], ['s', 'f'], ['q', 'f'], ['scal', 'f']], `if (c >= prm.nx*prm.nz) {return;} let a = scal[2]; p[c] += a*s[c]; r[c] -= a*q[c];`),
    updS: K([['s', 'f'], ['z', 'f'], ['scal', 'f']], `if (c >= prm.nx*prm.nz) {return;} s[c] = z[c] + scal[3]*s[c];`),
    projU: K([['typ', 'u'], ['p', 'f'], ['u', 'f'], ['F', 'f'], ['BU', 'f']], `let W = prm.nx+1u; if (c >= W*prm.nz) {return;} let i = c % W; let k = c / W;
      if (i == 0u || i == prm.nx) { return; } let cl = i-1u+prm.nx*k; let cr = i+prm.nx*k; let l = typ[cl]; let r = typ[cr];
      if (l == 0u || r == 0u || (l != 1u && r != 1u)) { return; }
      var th = 1.0; if (l == 2u || r == 2u) { th = clamp(F[cl] + F[cr] - 0.5, 0.1, 1.0); }
      u[c] = BU[c]*(u[c] - prm.dt/prm.dx*(p[cr] - p[cl])/th);`),
    projW: K([['typ', 'u'], ['p', 'f'], ['w', 'f'], ['F', 'f'], ['BW', 'f']], `let N = prm.nx; if (c >= N*(prm.nz+1u)) {return;} let i = c % N; let k = c / N;
      if (k == 0u || k == prm.nz) { return; } let cb = i+N*(k-1u); let ct = i+N*k; let b = typ[cb]; let t = typ[ct];
      if (b == 0u || t == 0u || (b != 1u && t != 1u)) { return; }
      var th = 1.0; if (b == 2u || t == 2u) { th = clamp(F[cb] + F[ct] - 0.5, 0.1, 1.0); }
      w[c] = BW[c]*(w[c] - prm.dt/prm.dx*(p[ct] - p[cb])/th);`),
    fluxX: K([['f', 'f'], ['u', 'f'], ['fl', 'f'], ['eta', 'f']], `let W = prm.nx+1u; if (c >= W*prm.nz) {return;} let i = i32(c % W); let k = i32(c / W);
      let vel = u[c]; if (vel == 0.0) { fl[c] = 0.0; return; } let s = abs(vel)*prm.dt/prm.dx;
      let di = select(i, i - 1, vel > 0.0);
      if (di < 0 && prm.mode < 0.5) {      // river inflow: the wet fraction of this face, below the inlet surface
        let e = max(eta[0], prm.zin + prm.dx);
        fl[c] = s*clamp((e - f32(k)*prm.dx)/prm.dx, 0.0, 1.0); return; }
      if (di < 0 || di >= i32(prm.nx)) { fl[c] = sign(vel)*s*fAt(di, k); return; }
      fl[c] = sign(vel)*flux1(di, k, vel, s, 0u);`, PLIC),
    fluxZ: K([['f', 'f'], ['w', 'f'], ['fl', 'f']], `let N = prm.nx; if (c >= N*(prm.nz+1u)) {return;} let i = i32(c % N); let k = i32(c / N);
      let vel = w[c]; if (vel == 0.0) { fl[c] = 0.0; return; } let s = abs(vel)*prm.dt/prm.dx;
      let dk = select(k, k - 1, vel > 0.0);
      if (dk < 0 || dk >= i32(prm.nz)) { fl[c] = 0.0; return; }
      fl[c] = sign(vel)*flux1(i, dk, vel, s, 1u);`, PLIC),
    updX: K([['f', 'f'], ['f0', 'f'], ['u', 'f'], ['fl', 'f'], ['solid', 'u']], `if (c >= prm.nx*prm.nz) {return;} if (solid[c] != 0u) {return;}
      let i = c % prm.nx; let k = c / prm.nx; let W = prm.nx+1u; let a = i + W*k;
      let cc = select(0.0, 1.0, f0[c] > 0.5);
      var v = f[c] - (fl[a+1u] - fl[a]) + cc*(u[a+1u] - u[a])*prm.dt/prm.dx;
      if (v < 1e-6) { v = 0.0; } if (v > 1.0 - 1e-6) { v = 1.0; } f[c] = v;`),
    updZ: K([['f', 'f'], ['f0', 'f'], ['w', 'f'], ['fl', 'f'], ['solid', 'u']], `if (c >= prm.nx*prm.nz) {return;} if (solid[c] != 0u) {return;}
      let N = prm.nx;
      let cc = select(0.0, 1.0, f0[c] > 0.5);
      var v = f[c] - (fl[c+N] - fl[c]) + cc*(w[c+N] - w[c])*prm.dt/prm.dx;
      if (v < 1e-6) { v = 0.0; } if (v > 1.0 - 1e-6) { v = 1.0; } f[c] = v;`),
    surface: K([['f', 'f'], ['solid', 'u'], ['eta', 'f']], `if (c >= prm.nx) {return;} let i = c; let N = prm.nx;
      var top = -1;
      for (var k = i32(prm.nz) - 1; k >= 0; k--) { let q = i + N*u32(k); if (solid[q] == 0u && f[q] > 0.5) { top = k; break; } }
      if (top < 0) { eta[i] = -1.0; return; }
      var above = 0.0; if (u32(top) + 1u < prm.nz) { above = f[i + N*(u32(top) + 1u)]; }
      eta[i] = (f32(top) + f[i + N*u32(top)] + above)*prm.dx;`),
    tick: K([['tb', 'f']], `if (c == 0u) { tb[0] += prm.dt; }`),
    // per column, for the measured section: discharge q (m2/s), water depth h (water volume per unit
    // width, so air pockets and overhangs count right) and the surface eta
    colQ: K([['u', 'f'], ['f', 'f'], ['solid', 'u'], ['eta', 'f'], ['egl', 'f']], `if (c >= prm.nx) {return;} let i = c; let N = prm.nx; let W = prm.nx + 1u;
      var q = 0.0; var h = 0.0;
      for (var k = 0u; k < prm.nz; k++) { let cl = i + N*k; if (solid[cl] != 0u) { continue; } let fw = clamp(f[cl], 0.0, 1.0);
        q += 0.5*(u[i + W*k] + u[i + 1u + W*k])*fw*prm.dx; h += fw*prm.dx; }
      egl[3u*i] = q; egl[3u*i + 1u] = h; egl[3u*i + 2u] = eta[i];`),
    // Smagorinsky eddy viscosity per wet cell, capped for explicit stability
    nut: K([['u', 'f'], ['w', 'f'], ['typ', 'u'], ['nt', 'f']], `if (c >= prm.nx*prm.nz) {return;} let i = c % prm.nx; let k = c / prm.nx; let W = prm.nx+1u; let N = prm.nx;
      if (typ[c] != 1u) { nt[c] = 0.0; return; }
      let dx = prm.dx;
      let dudx = (u[i+1u+W*k] - u[i+W*k])/dx; let dwdz = (w[i+N*(k+1u)] - w[i+N*k])/dx;
      let kd = select(k, k - 1u, k > 0u); let ku = min(k + 1u, prm.nz - 1u);
      let id = select(i, i - 1u, i > 0u); let iu = min(i + 1u, prm.nx - 1u);
      let ucu = 0.5*(u[i+W*ku] + u[i+1u+W*ku]); let ucd = 0.5*(u[i+W*kd] + u[i+1u+W*kd]);
      let wcr = 0.5*(w[iu+N*k] + w[iu+N*(k+1u)]); let wcl = 0.5*(w[id+N*k] + w[id+N*(k+1u)]);
      let dudz = (ucu - ucd)/(f32(ku - kd)*dx + 1e-9); let dwdx = (wcr - wcl)/(f32(iu - id)*dx + 1e-9);
      let S = sqrt(2.0*dudx*dudx + 2.0*dwdz*dwdz + (dudz + dwdx)*(dudz + dwdx));
      nt[c] = min(prm.cs*prm.cs*dx*dx*S, 0.2*dx*dx/prm.dt);`),
    // viscous + eddy diffusion of u (walls are handled by the wall-friction kernels, not here)
    diffU: K([['u', 'f'], ['typ', 'u'], ['nt', 'f'], ['o', 'f']], `let W = prm.nx+1u; if (c >= W*prm.nz) {return;} let i = c % W; let k = c / W; let N = prm.nx;
      if (i == 0u || i == prm.nx) { o[c] = u[c]; return; }
      let l = typ[i-1u+N*k]; let r = typ[i+N*k];
      if (l == 0u || r == 0u || (l != 1u && r != 1u)) { o[c] = u[c]; return; }
      let nu = prm.nu + 0.5*(nt[i-1u+N*k] + nt[i+N*k]);
      var lap = 0.0; let uc = u[c];
      if (i > 1u && typ[i-2u+N*k] != 0u) { lap += u[c-1u] - uc; }
      if (i < prm.nx-1u && typ[i+1u+N*k] != 0u) { lap += u[c+1u] - uc; }
      if (k > 0u && typ[i-1u+N*(k-1u)] != 0u && typ[i+N*(k-1u)] != 0u) { lap += u[c-W] - uc; }
      if (k < prm.nz-1u && typ[i-1u+N*(k+1u)] != 0u && typ[i+N*(k+1u)] != 0u) { lap += u[c+W] - uc; }
      o[c] = uc + prm.dt*nu*lap/(prm.dx*prm.dx);`),
    diffW: K([['w', 'f'], ['typ', 'u'], ['nt', 'f'], ['o', 'f']], `let N = prm.nx; if (c >= N*(prm.nz+1u)) {return;} let i = c % N; let k = c / N;
      if (k == 0u || k == prm.nz) { o[c] = w[c]; return; }
      let b = typ[i+N*(k-1u)]; let t = typ[i+N*k];
      if (b == 0u || t == 0u || (b != 1u && t != 1u)) { o[c] = w[c]; return; }
      let nu = prm.nu + 0.5*(nt[i+N*(k-1u)] + nt[i+N*k]);
      var lap = 0.0; let wc = w[c];
      if (k > 1u && typ[i+N*(k-2u)] != 0u) { lap += w[c-N] - wc; }
      if (k < prm.nz-1u && typ[i+N*(k+1u)] != 0u) { lap += w[c+N] - wc; }
      if (i > 0u && typ[i-1u+N*(k-1u)] != 0u && typ[i-1u+N*k] != 0u) { lap += w[c-1u] - wc; }
      if (i < N-1u && typ[i+1u+N*(k-1u)] != 0u && typ[i+1u+N*k] != 0u) { lap += w[c+1u] - wc; }
      o[c] = wc + prm.dt*nu*lap/(prm.dx*prm.dx);`),
    // implicit drag: porous (Darcy-Forchheimer) and plants (cell arrays A, B per direction),
    // plus log-law friction on faces that run along a wall: u /= 1 + dt (A + B |u|)
    dragU: K([['u', 'f'], ['solid', 'u'], ['dg', 'f'], ['ks', 'f'], ['BU', 'f'], ['pf', 'f']], `let W = prm.nx+1u; if (c >= W*prm.nz) {return;} let i = c % W; let k = c / W; let N = prm.nx;
      BU[c] = 1.0;
      if (i == 0u || i == prm.nx) { return; }
      let l = i-1u+N*k; let r = i+N*k; if (solid[l] != 0u || solid[r] != 0u) { return; }
      let A = 0.5*(dg[2u*l] + dg[2u*r]); var B = 0.5*(dg[2u*l+1u]*pf[l] + dg[2u*r+1u]*pf[r]);
      var kw = 0.0;
      if (k > 0u) { let a = l - N; let b = r - N; if (solid[a] != 0u) { kw = max(kw, ks[a]); } if (solid[b] != 0u) { kw = max(kw, ks[b]); } }
      if (k < prm.nz-1u) { let a = l + N; let b = r + N; if (solid[a] != 0u) { kw = max(kw, ks[a]); } if (solid[b] != 0u) { kw = max(kw, ks[b]); } }
      if (kw > 0.0) { let cf = 0.41/max(log(15.0*prm.dx/kw), 2.0); B += cf*cf/prm.dx; }
      BU[c] = 1.0/(1.0 + prm.dt*(A + B*abs(u[c])));`),
    dragW: K([['w', 'f'], ['solid', 'u'], ['dg', 'f'], ['ks', 'f'], ['BW', 'f']], `let N = prm.nx; if (c >= N*(prm.nz+1u)) {return;} let i = c % N; let k = c / N;
      BW[c] = 1.0;
      if (k == 0u || k == prm.nz) { return; }
      let b = i+N*(k-1u); let t = i+N*k; if (solid[b] != 0u || solid[t] != 0u) { return; }
      // gravity goes in after diffusion and drag: diffusing the gravity increment next to a wall
      // would leave a rotational part the pressure solve cannot remove (spurious currents)
      let A = 0.5*(dg[2u*b] + dg[2u*t]); var B = 0.5*(dg[2u*b+1u] + dg[2u*t+1u]);
      var kw = 0.0;
      if (i > 0u) { if (solid[b-1u] != 0u) { kw = max(kw, ks[b-1u]); } if (solid[t-1u] != 0u) { kw = max(kw, ks[t-1u]); } }
      if (i < N-1u) { if (solid[b+1u] != 0u) { kw = max(kw, ks[b+1u]); } if (solid[t+1u] != 0u) { kw = max(kw, ks[t+1u]); } }
      if (kw > 0.0) { let cf = 0.41/max(log(15.0*prm.dx/kw), 2.0); B += cf*cf/prm.dx; }
      BW[c] = 1.0/(1.0 + prm.dt*(A + B*abs(w[c])));
      w[c] = w[c] - prm.g*prm.dt;`),
  };

  // Water renderer: a full-screen triangle; each pixel finds its world point and samples f.
  const RENDER = `
struct V { x0:f32, z0:f32, s:f32, ve:f32, cw:f32, ch:f32, dx:f32, L:f32, nx:f32, nz:f32, mode:f32, H:f32, time:f32, umax:f32, depthRef:f32, pad:f32 };
@group(0) @binding(0) var<uniform> V0: V;
@group(0) @binding(1) var<storage, read> f: array<f32>;
@group(0) @binding(2) var<storage, read> solid: array<u32>;
@group(0) @binding(3) var<storage, read> u: array<f32>;
@group(0) @binding(4) var<storage, read> w: array<f32>;
@group(0) @binding(5) var<storage, read> eta: array<f32>;
@group(0) @binding(6) var<storage, read> dye: array<f32>;
@group(0) @binding(7) var<storage, read> cs: array<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
}
// local surface above row k in column i: (height, found) from the first non-solid air cell upward
// (eta is the topmost water in the column, so spray above would make the colour jump between columns)
fn surfAt(i:i32, k:i32) -> vec2f {
  let nx = i32(V0.nx); let nz = i32(V0.nz); var kk = k;
  // single cells for the first 16, then every second cell up to 64 (deep water changes colour slowly)
  for (var q = 0; q < 40; q++) {
    if (kk >= nz) { return vec2f(f32(nz)*V0.dx, 1.0); }
    let c = u32(i + nx*kk); let ca = u32(i + nx*min(kk + 1, nz - 1));
    // an air cell with water above it is a trapped pocket, not the surface: look past it
    if (solid[c] == 0u && f[c] < 0.5 && (kk + 1 >= nz || f[ca] < 0.5 || solid[ca] != 0u)) { return vec2f((f32(kk) + clamp(f[c], 0.0, 0.5))*V0.dx, 1.0); }
    kk += select(2, 1, q < 16);
  }
  return vec2f(0.0, 0.0);
}
fn smp(i:i32, k:i32) -> f32 {
  let nx = i32(V0.nx); let nz = i32(V0.nz);
  let ii = clamp(i, 0, nx - 1); let kk = clamp(k, 0, nz - 1); let c = u32(ii + nx*kk);
  return select(f[c], -1.0, solid[c] != 0u);
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let x = V0.x0 + pos.x / V0.s;
  let z = V0.z0 + (V0.ch - pos.y) / (V0.s * V0.ve);
  let nx = i32(V0.nx); let nz = i32(V0.nz);
  let gx = x / V0.dx - 0.5; let gz = z / V0.dx - 0.5;
  let i0 = i32(floor(gx)); let k0 = i32(floor(gz)); let a = gx - f32(i0); let b = gz - f32(k0);
  var s00 = smp(i0, k0); var s10 = smp(i0 + 1, k0); var s01 = smp(i0, k0 + 1); var s11 = smp(i0 + 1, k0 + 1);
  // a wall next to the water takes the water's value, so walls leave no dark seam
  var m = 0.0; var n = 0.0;
  if (s00 >= 0.0) { m += s00; n += 1.0; } if (s10 >= 0.0) { m += s10; n += 1.0; }
  if (s01 >= 0.0) { m += s01; n += 1.0; } if (s11 >= 0.0) { m += s11; n += 1.0; }
  m = select(0.0, m / n, n > 0.0);
  if (s00 < 0.0) { s00 = m; } if (s10 < 0.0) { s10 = m; } if (s01 < 0.0) { s01 = m; } if (s11 < 0.0) { s11 = m; }
  let fv = mix(mix(s00, s10, a), mix(s01, s11, a), b);
  let dfv = max(fwidth(fv), 1e-4);
  let ci = clamp(i32(x / V0.dx), 0, nx - 1); let ck = clamp(i32(z / V0.dx), 0, nz - 1);
  let cc = u32(ci + nx*ck);
  let inside = x >= 0.0 && x <= V0.L && z >= 0.0 && z <= V0.H && solid[cc] == 0u;
  let alpha = clamp((fv - 0.5) / dfv + 0.5, 0.0, 1.0);
  // colour by depth under the local surface (blended between columns, so no stripes)
  let gxe = clamp(x / V0.dx - 0.5, 0.0, V0.nx - 1.0); let ie0 = u32(floor(gxe)); let ie1 = min(ie0 + 1u, u32(nx - 1));
  let e0 = eta[ie0]; let e1 = eta[ie1]; let ae = gxe - f32(ie0);
  let e = select(select(mix(e0, e1, ae), e0, e1 < 0.0), e1, e0 < 0.0);
  // depth under the surface measured locally: up from this row to the first air cell (at most 64
  // cells) in the two nearest columns, with that cell's fill as the fraction, blended across, so spray
  // or a broken surface above does not make the colour jump and there are no column stripes
  let gxs = clamp(x / V0.dx - 0.5, 0.0, V0.nx - 1.0); let cs0 = i32(floor(gxs)); let cs1 = min(cs0 + 1, nx - 1);
  let sf0 = surfAt(cs0, ck); let sf1 = surfAt(cs1, ck); let as_ = gxs - f32(cs0);
  var zs = max(e, z + 64.0*V0.dx);
  if (sf0.y > 0.5 && sf1.y > 0.5) { zs = mix(sf0.x, sf1.x, as_); } else if (sf0.y > 0.5) { zs = sf0.x; } else if (sf1.y > 0.5) { zs = sf1.x; }
  let d = max(zs - z, 0.0);
  let t = clamp(d / V0.depthRef, 0.0, 1.0);
  // deep water dark navy, glowing brighter toward the surface (the cut face of a flume, lit from above)
  var col = mix(vec3f(0.08, 0.40, 0.66), vec3f(0.01, 0.05, 0.14), pow(t, 0.6));
  if (V0.mode > 0.5) {
    let W = u32(nx + 1);
    let uc = 0.5*(u[u32(ci) + W*u32(ck)] + u[u32(ci) + 1u + W*u32(ck)]);
    let wc = 0.5*(w[cc] + w[cc + u32(nx)]);
    let sp = clamp(length(vec2f(uc, wc)) / max(V0.umax, 1e-3), 0.0, 1.0);
    let c1 = mix(vec3f(0.05, 0.12, 0.35), vec3f(0.10, 0.65, 0.80), clamp(sp*2.0, 0.0, 1.0));
    col = mix(c1, vec3f(1.0, 0.86, 0.45), clamp(sp*2.0 - 1.0, 0.0, 1.0));
  }
  // suspended sediment: a sandy cloud by volume concentration (all classes)
  let sc = cs[3u*cc] + cs[3u*cc + 1u] + cs[3u*cc + 2u];
  col = mix(col, vec3f(0.62, 0.50, 0.33), clamp(sqrt(sc/0.004), 0.0, 0.85));
  // dye
  let dv = dye[cc];
  col = mix(col, vec3f(1.0, 0.62, 0.14), clamp(dv*1.4, 0.0, 0.92));     // amber (pink is the section cut)
  // (foam is drawn by the particles; a per-cell tint here showed the grid as squares)
  // bright line along the free surface
  let hl = 1.0 - clamp(abs(fv - 0.5) / (dfv*1.6), 0.0, 1.0);
  col = mix(col, vec3f(0.80, 0.95, 1.0), hl*0.7);
  let a1 = select(0.0, alpha*0.90, inside);
  return vec4f(col*a1, a1);
}
`;

  // Particles: instanced streaks from each particle back along its velocity, drawn additively.
  const RENDER_P = `
struct V { x0:f32, z0:f32, s:f32, ve:f32, cw:f32, ch:f32, dx:f32, L:f32, nx:f32, nz:f32, mode:f32, H:f32, time:f32, umax:f32, depthRef:f32, dpr:f32 };
@group(0) @binding(0) var<uniform> V0: V;
@group(0) @binding(1) var<storage, read> P: array<f32>;
struct VO { @builtin(position) pos: vec4f, @location(0) col: vec4f, @location(1) t: f32 };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let o = 8u*ii; let x = P[o]; let z = P[o+1u]; let vx = P[o+4u]; let vz = P[o+5u]; let age = P[o+6u]; let kind = P[o+7u];
  var out: VO;
  if (age < 0.0) { out.pos = vec4f(2.0, 2.0, 0.0, 1.0); out.col = vec4f(0.0); out.t = 0.0; return out; }
  let hx = (x - V0.x0)*V0.s; let hy = V0.ch - (z - V0.z0)*V0.s*V0.ve;
  let foam = kind > 0.5;
  let trail = select(0.06, 0.05, foam);                       // seconds of path behind the head
  var tx = hx - vx*trail*V0.s; var ty = hy + vz*trail*V0.s*V0.ve;
  var d = vec2f(hx - tx, hy - ty); var len = length(d);
  let maxLen = select(10.0, 6.0, foam)*V0.dpr;
  if (len > maxLen) { d = d*(maxLen/len); tx = hx - d.x; ty = hy - d.y; len = maxLen; }
  let minLen = 1.5*V0.dpr;
  if (len < minLen) { d = vec2f(minLen, 0.0); tx = hx - minLen; ty = hy; len = minLen; }
  let dir = d/len; let nrm = vec2f(-dir.y, dir.x);
  let hw = select(0.6, 0.9, foam)*V0.dpr;
  let corners = array<vec2f, 6>(vec2f(0.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(0.0, -1.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0));
  let cc = corners[vi];
  let p = mix(vec2f(tx, ty), vec2f(hx, hy), cc.x) + nrm*hw*cc.y;
  out.pos = vec4f(p.x/V0.cw*2.0 - 1.0, 1.0 - p.y/V0.ch*2.0, 0.0, 1.0);
  out.t = cc.x;
  if (!foam) {
    let sp = clamp(length(vec2f(vx, vz))/max(V0.umax, 0.05), 0.0, 1.0);
    out.col = vec4f(mix(vec3f(0.20, 0.45, 0.80), vec3f(0.55, 0.80, 1.0), sp), 0.04 + 0.12*sp);   // a faint texture, not loops
  } else {
    out.col = vec4f(1.0, 1.0, 1.0, 0.9*clamp(1.0 - age/3.0, 0.0, 1.0));
  }
  return out;
}
@fragment fn fs(in: VO) -> @location(0) vec4f { let a = in.col.a*in.t; return vec4f(in.col.rgb*a, a); }
`;

  // Long exposure (after Prandtl's and Van Dyke's flake photographs): every frame the exposure image
  // fades by exp(-dt/T) and each particle's step is added as a thin line, so slow water, where flakes
  // linger, exposes brighter, and fast water draws long faint streaks. T is the exposure time.
  const FS_TRI = `@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { let p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0)); return vec4f(p[i], 0.0, 1.0); }`;
  const EXP_FADE = FS_TRI + `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> S: vec4f;
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f { return textureLoad(src, vec2i(pos.xy), 0) * S.x; }`;
  const EXP_LINE = `
struct V { x0:f32, z0:f32, s:f32, ve:f32, cw:f32, ch:f32, dx:f32, L:f32, nx:f32, nz:f32, mode:f32, H:f32, time:f32, umax:f32, depthRef:f32, dpr:f32 };
@group(0) @binding(0) var<uniform> V0: V;
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<uniform> S: vec4f;
struct VO { @builtin(position) pos: vec4f, @location(0) a: f32 };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let o = 8u*ii; var out: VO; out.a = 0.0;
  if (P[o+6u] < 0.0) { out.pos = vec4f(2.0, 2.0, 0.0, 1.0); return out; }
  let hx = (P[o] - V0.x0)*V0.s; let hy = V0.ch - (P[o+1u] - V0.z0)*V0.s*V0.ve;
  let tx = (P[o+2u] - V0.x0)*V0.s; let ty = V0.ch - (P[o+3u] - V0.z0)*V0.s*V0.ve;
  var d = vec2f(hx - tx, hy - ty); var len = length(d);
  if (len > 60.0*V0.dpr) { out.pos = vec4f(2.0, 2.0, 0.0, 1.0); return out; }
  let mn = 0.75*V0.dpr; var dir = vec2f(1.0, 0.0); if (len > 1e-3) { dir = d/len; }
  let ext = max(len, mn); let nrm = vec2f(-dir.y, dir.x); let hw = 0.5*V0.dpr;
  let corners = array<vec2f, 6>(vec2f(0.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(0.0, -1.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0));
  let cc = corners[vi];
  let p = vec2f(tx, ty) + dir*ext*cc.x + nrm*hw*cc.y;
  out.pos = vec4f(p.x/V0.cw*2.0 - 1.0, 1.0 - p.y/V0.ch*2.0, 0.0, 1.0);
  out.a = S.y * mn / ext;          // the same light per step, spread over the length it travelled
  return out;
}
@fragment fn fs(in: VO) -> @location(0) vec4f { return vec4f(in.a, in.a, in.a, in.a); }`;
  const EXP_SHOW = FS_TRI + `
@group(0) @binding(0) var acc: texture_2d<f32>;
@group(0) @binding(1) var<uniform> S: vec4f;
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let a = textureLoad(acc, vec2i(pos.xy), 0).r;
  let l = 1.0 - exp(-a*S.z);
  let col = mix(vec3f(0.45, 0.72, 1.0), vec3f(0.95, 0.98, 1.0), l)*l*0.95;
  return vec4f(col, l*0.95);
}`;

  // smoke comb filaments: a segment from each particle to the one released before it on the same ring;
  // pulse marks (every Pth release) drawn brighter and wider; older smoke fades
  const COMB_LINE = `
struct V { x0:f32, z0:f32, s:f32, ve:f32, cw:f32, ch:f32, dx:f32, L:f32, nx:f32, nz:f32, mode:f32, H:f32, time:f32, umax:f32, depthRef:f32, dpr:f32 };
@group(0) @binding(0) var<uniform> V0: V;
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<uniform> C: vec4f;          // newest slot, M, P, -
struct VO { @builtin(position) pos: vec4f, @location(0) a: f32 };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  var out: VO; out.a = 0.0; out.pos = vec4f(2.0, 2.0, 0.0, 1.0);
  let Mm = u32(C.y); let j = ii % Mm; let n = u32(C.x);
  if (j == (n + 1u) % Mm) { return out; }
  let jp = select(j - 1u, Mm - 1u, j == 0u); let ip = ii - j + jp;
  let o = 4u*ii; let op = 4u*ip;
  if (P[o+2u] < 0.0 || P[op+2u] < 0.0) { return out; }
  let ax = (P[o] - V0.x0)*V0.s; let ay = V0.ch - (P[o+1u] - V0.z0)*V0.s*V0.ve;
  let bx = (P[op] - V0.x0)*V0.s; let by = V0.ch - (P[op+1u] - V0.z0)*V0.s*V0.ve;
  let d = vec2f(bx - ax, by - ay); let len = length(d);
  if (len > 14.0*V0.dpr) { return out; }          // torn apart: the smoke has dispersed
  var dir = vec2f(1.0, 0.0); if (len > 1e-4) { dir = d/len; }
  let pulse = (u32(P[o+3u]) % u32(C.z)) == 0u;
  // a stretched filament thins (the same smoke over a longer line), so it fades with its stretch
  let thin = clamp(5.0*V0.dpr/max(len, 1e-3), 0.0, 1.0);
  let hw = select(0.55, 1.1, pulse)*V0.dpr; let nrm = vec2f(-dir.y, dir.x);
  let corners = array<vec2f, 6>(vec2f(0.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(0.0, -1.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0));
  let cc = corners[vi]; let ext = max(len, select(0.5, 2.0, pulse)*V0.dpr);
  let p = vec2f(ax, ay) + dir*ext*cc.x + nrm*hw*cc.y;
  out.pos = vec4f(p.x/V0.cw*2.0 - 1.0, 1.0 - p.y/V0.ch*2.0, 0.0, 1.0);
  let older = f32((n + Mm - j) % Mm)/f32(Mm);     // 0 newest .. 1 oldest
  out.a = select(0.62, 1.0, pulse)*(1.0 - smoothstep(0.35, 1.0, older))*pow(thin, 1.5);
  return out;
}
@fragment fn fs(in: VO) -> @location(0) vec4f { let c = vec3f(0.93, 0.96, 1.0)*in.a; return vec4f(c, in.a); }`;

  // bubble wire timelines: a segment from each mark to the next one up the same row; brightest when
  // just released, fading as the row ages
  const WIRE_LINE = `
struct V { x0:f32, z0:f32, s:f32, ve:f32, cw:f32, ch:f32, dx:f32, L:f32, nx:f32, nz:f32, mode:f32, H:f32, time:f32, umax:f32, depthRef:f32, dpr:f32 };
@group(0) @binding(0) var<uniform> V0: V;
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<uniform> C: vec4f;          // newest slot, R, K, -
struct VO { @builtin(position) pos: vec4f, @location(0) a: f32 };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  var out: VO; out.a = 0.0; out.pos = vec4f(2.0, 2.0, 0.0, 1.0);
  let R = u32(C.y); let Kk = u32(C.z); let k = ii % Kk; let r = (ii / Kk) % R;
  if (k + 1u >= Kk) { return out; }
  let o = 4u*ii; let op = 4u*(ii + 1u);
  if (P[o+2u] < 0.0 || P[op+2u] < 0.0) { return out; }
  let ax = (P[o] - V0.x0)*V0.s; let ay = V0.ch - (P[o+1u] - V0.z0)*V0.s*V0.ve;
  let bx = (P[op] - V0.x0)*V0.s; let by = V0.ch - (P[op+1u] - V0.z0)*V0.s*V0.ve;
  let d = vec2f(bx - ax, by - ay); let len = length(d);
  if (len > 40.0*V0.dpr) { return out; }
  var dir = vec2f(0.0, -1.0); if (len > 1e-4) { dir = d/len; }
  let hw = 0.6*V0.dpr; let nrm = vec2f(-dir.y, dir.x);
  let corners = array<vec2f, 6>(vec2f(0.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(0.0, -1.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0));
  let cc = corners[vi]; let ext = max(len, 0.5*V0.dpr);
  let p = vec2f(ax, ay) + dir*ext*cc.x + nrm*hw*cc.y;
  out.pos = vec4f(p.x/V0.cw*2.0 - 1.0, 1.0 - p.y/V0.ch*2.0, 0.0, 1.0);
  let older = f32((u32(C.x) + R - r) % R)/f32(R);
  out.a = 0.9*pow(1.0 - older, 1.3);
  return out;
}
@fragment fn fs(in: VO) -> @location(0) vec4f { let c = vec3f(0.95, 0.97, 1.0)*in.a; return vec4f(c, in.a); }`;

  // ───────────────────────── wave theory ─────────────────────────
  function waveNumber(omega, h) {           // omega^2 = g k tanh(k h)
    let k = omega * omega / G; if (h <= 0) return k;
    for (let n = 0; n < 50; n++) {
      const th = Math.tanh(k * h), F = G * k * th - omega * omega;
      const dF = G * th + G * k * h * (1 - th * th);
      const dk = F / dF; k -= dk; if (Math.abs(dk) < 1e-12 * k) break;
    }
    return k;
  }
  function pistonStroke(H, T, h) {          // linear piston wavemaker: H/S = 2(cosh 2kh - 1)/(sinh 2kh + 2kh)
    const om = 2 * Math.PI / T, k = waveNumber(om, h), kh = k * h;
    const ratio = 2 * (Math.cosh(2 * kh) - 1) / (Math.sinh(2 * kh) + 2 * kh);
    return { S: H / ratio, k, L: 2 * Math.PI / k, om };
  }

  // ───────────────────────── model ─────────────────────────
  const DEFAULTS = {
    lab:   { type: 'river', h0: 0.40, q: 0.10, hout: 0.40, H: 0.08, T: 2.0, speed: 1, farEnd: 'beach', absorb: true, sediment: true, morfac: 10, feed: true },
    field: { type: 'river', h0: 6.0, q: 6.0, hout: 6.0, H: 1.5, T: 8.0, speed: 1, farEnd: 'beach', absorb: true, sediment: true, morfac: 10, feed: true },
  };

  const M = {
    id: 'flume', title: 'Flume',
    gpu: null, device: null, ctx: null, format: null, grid: null, ok: false, reason: '',
    p: Object.assign({}, DEFAULTS.lab), scale: 'lab',
    t: 0, dt: 0.005, vmax: 0, vol: 0, vol0: 0, msStep: 0, stepsFrame: 0, busy: false, readBusy: false, frameNo: 0,
    colorMode: 0, api: null,
  };

  M.domain = function (scale) {
    return scale === 'field' ? { L: 400, H: 16, dx: 0.2 } : { L: 20, H: 1.0, dx: 0.01 };
  };

  M.controls = {
    html: `
      <div class="row"><label for="m-type">flume</label>
        <select id="m-type"><option value="river">river (inflow)</option><option value="wave">wave (paddle)</option></select></div>
      <div class="row"><label for="m-h0">water level (m)</label><input id="m-h0" type="number" step="0.01" min="0"></div>
      <div class="row m-river"><label for="m-q">discharge (m²/s per m)</label><input id="m-q" type="number" step="0.01" min="0"></div>
      <div class="row m-river"><label for="m-hout">tailwater depth (m)</label><input id="m-hout" type="number" step="0.01" min="0"></div>
      <div class="row m-wave"><label for="m-wh">wave height (m)</label><input id="m-wh" type="number" step="0.01" min="0"></div>
      <div class="row m-wave"><label for="m-wt">wave period (s)</label><input id="m-wt" type="number" step="0.1" min="0.3"></div>
      <div class="row m-wave"><label for="m-far">far end</label><select id="m-far"><option value="beach">absorbing</option><option value="wall">wall (reflects)</option></select></div>
      <div class="row m-wave"><label for="m-abs">paddle absorbs</label><select id="m-abs"><option value="1">yes</option><option value="0">no</option></select></div>
      <div class="row"><label for="m-sed">moving bed</label><select id="m-sed"><option value="1">on</option><option value="0">off</option></select></div>
      <div class="row m-river"><label for="m-feed">sediment supply</label><select id="m-feed"><option value="1">equilibrium</option><option value="0">none (clear water)</option></select></div>
      <div class="row"><label for="m-mf">bed speed-up</label><input id="m-mf" type="number" min="1" max="200" step="1" title="the bed changes this many times faster than the water (morphological factor)"></div>
      <div class="row"><label for="m-part">particles</label><select id="m-part"><option value="both">tracers and foam</option><option value="foam">foam only</option><option value="tracers">tracers only</option><option value="off">off</option></select></div>
      <div class="row"><label for="m-color">water colour</label>
        <select id="m-color"><option value="0">depth</option><option value="1">speed</option></select></div>`,
    wire() {
      const $ = (id) => document.getElementById(id);
      const map = { 'm-h0': 'h0', 'm-q': 'q', 'm-hout': 'hout', 'm-wh': 'H', 'm-wt': 'T' };
      for (const id in map) $(id).addEventListener('change', (e) => {
        const v = parseFloat(e.target.value); if (!isFinite(v) || v < 0) { M.syncControls(); return; }
        M.p[map[id]] = v; M.applyParams(); if (id === 'm-h0') M.reset(); M.api && M.api.changed();
      });
      $('m-type').addEventListener('change', (e) => { M.p.type = e.target.value; M.syncControls(); M.applyParams(); M.reset(); M.api && M.api.changed(); });
      $('m-color').addEventListener('change', (e) => { M.colorMode = +e.target.value; });
      $('m-part').addEventListener('change', (e) => { M.particles = e.target.value !== 'off'; M.drawKinds = e.target.value; });
      $('m-sed').addEventListener('change', (e) => { M.p.sediment = e.target.value === '1'; M.applyParams(); M.api && M.api.changed(); });
      $('m-feed').addEventListener('change', (e) => { M.p.feed = e.target.value === '1'; M.applyParams(); M.api && M.api.changed(); });
      $('m-mf').addEventListener('change', (e) => { const v = parseFloat(e.target.value); if (isFinite(v) && v >= 1) M.p.morfac = Math.min(v, 200); M.syncControls(); M.applyParams(); M.api && M.api.changed(); });
      $('m-far').addEventListener('change', (e) => { M.p.farEnd = e.target.value; M.applyParams(); M.api && M.api.changed(); });
      $('m-abs').addEventListener('change', (e) => { M.p.absorb = e.target.value === '1'; M.applyParams(); M.api && M.api.changed(); });
      M.syncControls();
    },
  };
  M.syncControls = function () {
    if (typeof document === 'undefined') return;
    const $ = (id) => document.getElementById(id), p = M.p;
    const set = (id, v) => { const el = $(id); if (el) el.value = String(+v.toFixed(4)); };
    if (!$('m-type')) return;
    $('m-sed').value = p.sediment === false ? '0' : '1'; $('m-feed').value = p.feed === false ? '0' : '1'; set('m-mf', p.morfac || 10);
    $('m-type').value = p.type; $('m-far').value = p.farEnd === 'wall' ? 'wall' : 'beach'; $('m-abs').value = p.absorb === false ? '0' : '1'; set('m-h0', p.h0); set('m-q', p.q); set('m-hout', p.hout); set('m-wh', p.H); set('m-wt', p.T);
    for (const el of document.querySelectorAll('.m-river')) el.style.display = p.type === 'river' ? '' : 'none';
    for (const el of document.querySelectorAll('.m-wave')) el.style.display = p.type === 'wave' ? '' : 'none';
  };

  M.state = function () { return Object.assign({}, M.p, { colorMode: M.colorMode }); };
  M.setState = function (s) {
    if (!s) return;
    for (const k of Object.keys(DEFAULTS.lab)) {
      if (s[k] == null) continue;
      if (k === 'type') { if (s[k] === 'river' || s[k] === 'wave') M.p.type = s[k]; }
      else if (k === 'farEnd') M.p.farEnd = s[k] === 'wall' ? 'wall' : 'beach';
      else if (k === 'absorb' || k === 'sediment' || k === 'feed') M.p[k] = !!s[k];
      else if (isFinite(+s[k])) M.p[k] = +s[k];
    }
    if (s.colorMode != null) M.colorMode = +s.colorMode;
    M.syncControls(); if (M.grid) M.applyParams();
  };
  M.setScale = function (scale) { M.scale = scale; M.p = Object.assign({}, DEFAULTS[scale] || DEFAULTS.lab, { type: M.p.type, farEnd: M.p.farEnd, absorb: M.p.absorb, sediment: M.p.sediment, morfac: M.p.morfac, feed: M.p.feed }); M.syncControls(); if (M.grid) M.applyParams(); };

  M.init = async function (api) {
    M.api = api || null;
    const gpu = (api && api.gpu) || (typeof navigator !== 'undefined' && navigator.gpu);
    if (!gpu) { M.reason = 'This browser has no WebGPU. The flume needs it to run (Chrome, Edge, or Safari 26 and later). You can still draw and save scenes.'; return { ok: false, reason: M.reason }; }
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) { M.reason = 'WebGPU is present but no GPU adapter was found.'; return { ok: false, reason: M.reason }; }
    M.adapter = adapter;                    // keep it referenced
    M.device = await adapter.requestDevice();
    M.device.addEventListener && M.device.addEventListener('uncapturederror', (e) => { console.error('flume GPU error:', e.error && e.error.message); });
    // A lost device (driver reset, GPU watchdog) is rebuilt and the flume restarts from its scene.
    const dev = M.device;
    if (dev.lost) dev.lost.then((info) => { if (M.device === dev && info.reason !== 'destroyed') M.recover(info.message); });
    const canvas = api && api.canvas;
    // read-back mode: render offscreen and copy the pixels into a 2D canvas (for headless
    // browsers that cannot present a WebGPU canvas, and for stills)
    M.readback = !!(api && api.readback);
    if (canvas && canvas.getContext && M.readback) { M.rbCtx = M.rbCtx || canvas.getContext('2d'); M.format = 'rgba8unorm'; }
    else if (canvas && canvas.getContext) {
      M.ctx = canvas.getContext('webgpu');
      M.format = navigator.gpu.getPreferredCanvasFormat();
      M.ctx.configure({ device: M.device, format: M.format, alphaMode: 'premultiplied' });
    }
    M.ok = true; return { ok: true };
  };

  M.recover = async function (why) {
    console.warn('flume: GPU device lost (' + (why || 'no reason given') + '), restarting');
    M.ok = false; M.S = null; M.render_ = null; pipes = {}; M.busy = false; M.readBusy = false;
    M.api && M.api.notify && M.api.notify('The GPU was reset. The flume restarted from your scene.');
    const r = await M.init(M.api);
    if (!r.ok) return;
    if (M.grid) { M.setGrid(M.grid, M.raster); M.reset(); }
  };

  // ── buffers and dispatches ──
  const SU = () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  let pipes = {};
  function pipe(name, constants) {
    const key = name + JSON.stringify(constants || {});
    if (!pipes[key]) {
      const module = M.device.createShaderModule({ code: SRC[name] });
      pipes[key] = M.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main', constants } });
    }
    return pipes[key];
  }
  function buf(n, data) {
    const b = M.device.createBuffer({ size: Math.max(16, Math.ceil(n * 4 / 16) * 16), usage: SU() });
    if (data) M.device.queue.writeBuffer(b, 0, data);
    return b;
  }
  function ubuf() { return M.device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); }
  function writeP(b, o) {
    const ab = new ArrayBuffer(96), u = new Uint32Array(ab), f = new Float32Array(ab);
    u[0] = o.nx; u[1] = o.nz; u[2] = o.cnx || 0; u[3] = o.cnz || 0;
    const fl = [o.dx, o.dt, o.g, o.mode, o.uin, o.hin, o.uout, o.hout, o.S, o.om, o.sx0, o.srate, o.zin, o.zout, o.q, o.nu, o.cs, o.amp, o.absorb, 0];
    for (let j = 0; j < fl.length; j++) f[4 + j] = fl[j] || 0;
    M.device.queue.writeBuffer(b, 0, ab);
  }
  function D(name, uni, bufs, count, constants, wg = 64) {
    const p = pipe(name, constants);
    const bg = M.device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: uni } }, ...bufs.map((b, j) => ({ binding: j + 1, resource: { buffer: b } }))] });
    return { p, bg, n: Math.ceil(count / wg) };
  }
  function run(pass, x) { pass.setPipeline(x.p); pass.setBindGroup(0, x.bg); pass.dispatchWorkgroups(x.n); }

  M.setGrid = function (grid, raster) {
    if (!M.ok) { M.grid = grid; return; }
    if (M.S) for (const b of M.S.all) b.destroy();
    const { nx, nz } = grid, N = nx * nz, NU = (nx + 1) * nz, NW = nx * (nz + 1);
    const B = {
      f: buf(N), f0: buf(N), solid: buf(N), typ: buf(N),
      u: buf(NU), u2: buf(NU), uvA: buf(NU), uvB: buf(NU), w: buf(NW), w2: buf(NW), wvA: buf(NW), wvB: buf(NW),
      flx: buf(Math.max(NU, NW)), p: buf(N), r: buf(N), z: buf(N), s: buf(N), q: buf(N), tmp: buf(N), nt: buf(N), dgX: buf(2 * N), dgZ: buf(2 * N), ks: buf(N), BU: buf(NU), BW: buf(NW), uM: buf(NU), wM: buf(NW),
      bed: buf(3 * nx), cell: buf(4 * N), colx: buf(6 * nx), qc: buf(3 * nx), qb: buf(3 * (nx + 1)), ust: buf(nx), ustK: buf(nx), pl: buf(5 * nx), pf: buf(N), po: buf(2 * nx), dye: buf(N), dye2: buf(N), P: buf(8 * (Math.max(3000, Math.min(20000, Math.round(N / 10))) + 16000)), PS: buf(8 * Math.max(20000, Math.min(150000, Math.round(N * 0.6)))), CB: buf(4 * COMB.S * COMB.M), WB: buf(4 * WIRE.W * WIRE.R * WIRE.K), cs: buf(3 * N), cs2: buf(3 * N),
      part: buf(Math.ceil(N / 256) + 1), scal: buf(8), tb: buf(4), eta: buf(nx), egl: buf(3 * nx),
    };
    const P0 = ubuf();
    const SPU = M.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const DIU = M.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const PU = M.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const PU2 = M.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const CU = M.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const WU = M.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const nS = Math.max(20000, Math.min(150000, Math.round(N * 0.6)));
    const nT = Math.max(3000, Math.min(20000, Math.round(N / 10))), nF = 16000;
    const L = [{ nx, nz, typ: B.typ, x: B.z, b: B.r, r: buf(N) }];
    while (L[L.length - 1].nx > 8 && L[L.length - 1].nz > 4) {
      const f = L[L.length - 1], cnx = Math.ceil(f.nx / 2), cnz = Math.ceil(f.nz / 2), n = cnx * cnz;
      L.push({ nx: cnx, nz: cnz, typ: buf(n), x: buf(n), b: buf(n), r: buf(n) });
    }
    const all = Object.values(B).concat(L.slice(1).flatMap(v => [v.typ, v.x, v.b, v.r]), [L[0].r]);
    for (let l = 0; l < L.length; l++) {
      const v = L[l]; v.U = ubuf(); writeP(v.U, { nx: v.nx, nz: v.nz });
      const n = v.nx * v.nz;
      if (l === 0) {
        v.red = D('rbgs0', v.U, [v.typ, v.x, v.b, B.f, B.BU, B.BW], n, { COLOR: 0 });
        v.black = D('rbgs0', v.U, [v.typ, v.x, v.b, B.f, B.BU, B.BW], n, { COLOR: 1 });
        v.res = D('resid0', v.U, [v.typ, v.x, v.b, v.r, B.f, B.BU, B.BW], n);
      } else {
        v.red = D('rbgs', v.U, [v.typ, v.x, v.b], n, { COLOR: 0 });
        v.black = D('rbgs', v.U, [v.typ, v.x, v.b], n, { COLOR: 1 });
        v.res = D('resid', v.U, [v.typ, v.x, v.b, v.r], n);
      }
      if (l + 1 < L.length) {
        const c = L[l + 1]; v.UC = ubuf(); writeP(v.UC, { nx: v.nx, nz: v.nz, cnx: c.nx, cnz: c.nz });
        v.coarsen = D('coarsen', v.UC, [v.typ, c.typ], c.nx * c.nz);
        v.restr = D('restrict', v.UC, [v.typ, v.r, c.typ, c.b], c.nx * c.nz);
        v.prol = D('prolong', v.UC, [v.typ, v.x, c.typ, c.x], n);
      }
    }
    const d = {
      classify: D('classify', P0, [B.f, B.solid, B.typ], N), clearSolid: D('clearSolid', P0, [B.f, B.solid], N),
      validU: D('validU', P0, [B.typ, B.uvA], NU), validW: D('validW', P0, [B.typ, B.wvA], NW),
      exUab: D('extrapU', P0, [B.u, B.uvA, B.u2, B.uvB], NU), exUba: D('extrapU', P0, [B.u2, B.uvB, B.u, B.uvA], NU),
      exWab: D('extrapW', P0, [B.w, B.wvA, B.w2, B.wvB], NW), exWba: D('extrapW', P0, [B.w2, B.wvB, B.w, B.wvA], NW),
      advU: D('advectU', P0, [B.u, B.w, B.u2], NU), advW: D('advectW', P0, [B.u, B.w, B.w2], NW),
      advFU: D('advFU', P0, [B.u, B.w, B.u2, B.typ], NU), advFW: D('advFW', P0, [B.u, B.w, B.w2, B.typ], NW),
      sedShear: D('sedShear', P0, [B.u, B.f, B.solid, B.bed, B.cell, B.qc, B.ust, SPU], nx),
      sedFlux: D('sedFlux', P0, [B.qc, B.bed, B.qb, SPU], nx + 1),
      sedExner: D('sedExner', P0, [B.qb, B.bed, B.colx, B.cell, SPU], nx),
      sedAvalR: D('sedAval', P0, [B.bed, SPU], Math.ceil(nx / 2), { PAR: 0 }),
      sedAvalB: D('sedAval', P0, [B.bed, SPU], Math.ceil(nx / 2), { PAR: 1 }),
      dyeAdv: D('dyeAdv', P0, [B.u, B.w, B.typ, B.dye, B.dye2], N),
      dyeInject: D('dyeInject', P0, [B.f, B.solid, B.dye, DIU], N),
      partStep: D('partStep', P0, [B.u, B.w, B.f, B.solid, B.eta, B.P, PU], nT + nF),
      partStepS: D('partStep', P0, [B.u, B.w, B.f, B.solid, B.eta, B.PS, PU2], nS),
      combStep: D('combStep', P0, [B.u, B.w, B.f, B.solid, B.eta, B.CB, CU], COMB.S * COMB.M),
      wireStep: D('wireStep', P0, [B.u, B.w, B.f, B.solid, B.eta, B.WB, WU], WIRE.W * WIRE.R * WIRE.K),
      plantBend: D('plantBend', P0, [B.u, B.pl, B.pf, B.po], nx),
      susAdv: D('susAdv', P0, [B.u, B.w, B.typ, B.cs, B.cs2, B.ustK, B.bed, B.eta, SPU], N),
      susBed: D('susBed', P0, [B.f, B.solid, B.bed, B.cell, B.cs, B.colx, B.ust, B.ustK, SPU], nx),
      sedCells: D('sedCells', P0, [B.solid, B.f, B.bed, B.cell, B.colx, B.tb, B.ks, B.cs, SPU], N),
      macU: D('macU', P0, [B.u, B.w, B.u2, B.uM, B.typ], NU), macW: D('macW', P0, [B.u, B.w, B.w2, B.wM, B.typ], NW),
      bcU: D('bcU', P0, [B.u, B.solid, B.tb, B.eta], NU), bcW: D('bcW', P0, [B.w, B.solid], NW),
      spongeU: D('spongeU', P0, [B.u], NU), spongeW: D('spongeW', P0, [B.w], NW),
      rhs: D('rhs', P0, [B.typ, B.u, B.w, B.r, B.p, B.BU, B.BW], N),
      applyA: D('applyA0', P0, [B.typ, B.s, B.q, B.f, B.BU, B.BW], N),
      dotSQ: D('dot', P0, [B.s, B.q, B.part], N, undefined, 256), finSQ: D('fin', P0, [B.part, B.scal], 256, { SLOT: 1, OP: 0 }, 256),
      dotZR: D('dot', P0, [B.z, B.r, B.part], N, undefined, 256),
      fin0: D('fin', P0, [B.part, B.scal], 256, { SLOT: 0, OP: 0 }, 256), fin4: D('fin', P0, [B.part, B.scal], 256, { SLOT: 4, OP: 0 }, 256),
      alpha: D('alpha', P0, [B.scal], 1), beta: D('beta', P0, [B.scal], 1),
      updPR: D('updPR', P0, [B.p, B.r, B.s, B.q, B.scal], N), updS: D('updS', P0, [B.s, B.z, B.scal], N),
      projU: D('projU', P0, [B.typ, B.p, B.u, B.f, B.BU], NU), projW: D('projW', P0, [B.typ, B.p, B.w, B.f, B.BW], NW),
      fluxX: D('fluxX', P0, [B.f, B.u, B.flx, B.eta], NU), fluxZ: D('fluxZ', P0, [B.f, B.w, B.flx], NW),
      updX: D('updX', P0, [B.f, B.f0, B.u, B.flx, B.solid], N), updZ: D('updZ', P0, [B.f, B.f0, B.w, B.flx, B.solid], N),
      surface: D('surface', P0, [B.f, B.solid, B.eta], nx), tick: D('tick', P0, [B.tb], 1),
      colQ: D('colQ', P0, [B.u, B.f, B.solid, B.eta, B.egl], nx),
      speed: D('speed', P0, [B.u, B.w, B.typ, B.tmp], N),
      nut: D('nut', P0, [B.u, B.w, B.typ, B.nt], N),
      diffU: D('diffU', P0, [B.u, B.typ, B.nt, B.u2], NU), diffW: D('diffW', P0, [B.w, B.typ, B.nt, B.w2], NW),
      dragU: D('dragU', P0, [B.u, B.solid, B.dgX, B.ks, B.BU, B.pf], NU), dragW: D('dragW', P0, [B.w, B.solid, B.dgZ, B.ks, B.BW], NW),
      redVol: D('red', P0, [B.f, B.part], N, { OP: 0 }, 256), finVol: D('fin', P0, [B.part, B.scal], 256, { SLOT: 6, OP: 0 }, 256),
      redMax: D('red', P0, [B.tmp, B.part], N, { OP: 1 }, 256), finMax: D('fin', P0, [B.part, B.scal], 256, { SLOT: 7, OP: 1 }, 256),
    };
    M.S = { B, L, d, P0, SPU, DIU, PU, PU2, CU, WU, nT, nF, nS, N, NU, NW, all: all.concat([P0, SPU, DIU, PU, PU2, CU, WU], L.flatMap(v => [v.U, v.UC].filter(Boolean))) };
    M.grid = grid;
    M.render_ = null;
    if (raster) M.setRaster(raster);
    M.applyParams();
  };

  M.setRaster = function (raster) {
    M.gen = (M.gen || 0) + 1;     // a design edit also outdates readbacks in flight
    M.raster = raster;
    if (!M.ok || !M.S) return;
    const N = M.S.N, solid = new Uint32Array(N), m = raster.mat;
    for (let c = 0; c < N; c++) solid[c] = (m[c] === 1 || m[c] === 4) ? 1 : 0;
    M.device.queue.writeBuffer(M.S.B.solid, 0, solid);
    const e = M.device.createCommandEncoder(), pass = e.beginComputePass();
    run(pass, M.S.d.clearSolid); run(pass, M.S.d.surface); pass.end(); M.device.queue.submit([e.finish()]);
    M.solidHost = solid;
    // drag per cell (A in 1/s, B in 1/m) for horizontal and vertical flow, and wall roughness ks (m)
    const dgX = new Float32Array(2 * N), dgZ = new Float32Array(2 * N), ks = new Float32Array(N), nu = NU_WATER;
    const ksSolid = M.scale === 'field' ? 0.02 : 0.001;
    for (let c = 0; c < N; c++) {
      if (m[c] === 2) {              // porous: van Gent (1995) form, superficial velocity
        const n = Math.min(Math.max(raster.p1[c] || 0.4, 0.05), 0.95), d = Math.max(raster.p2[c] || 0.04, 1e-4);
        const A = POROUS_ALPHA * (1 - n) * (1 - n) / (n * n * n) * nu / (d * d), B = POROUS_BETA * (1 - n) / (n * n * n) / d;
        dgX[2 * c] = dgZ[2 * c] = A; dgX[2 * c + 1] = dgZ[2 * c + 1] = B;
      } else if (m[c] === 3) {       // plants: rigid stems, drag on horizontal flow only
        dgX[2 * c + 1] = 0.5 * (raster.p2[c] || 1) * (raster.p1[c] || 8);
      } else if (m[c] === 4) ks[c] = 2.5 * (GRAIN_D[raster.cls[c]] || GRAIN_D[0]);
      else if (m[c] === 1) ks[c] = ksSolid;
    }
    if (M.wallFriction === false) ks.fill(0);       // for tests: walls and bed without friction
    // the erodible bed per column: the sediment under the lowest water cell, down to the top of any
    // fixed solid (zfix), up to the next solid above (zcap)
    const { nx, nz, dx } = M.grid, zs = new Float32Array(nx), zfix = new Float32Array(nx), zcap = new Float32Array(nx);
    const comp = new Float32Array(3 * N), tdep = new Float32Array(N).fill(-1);
    for (let c = 0; c < N; c++) if (m[c] === 4) comp[3 * c + Math.min(2, raster.cls[c] | 0)] = 1;
    for (let i = 0; i < nx; i++) {
      let kf = 0; while (kf < nz && (m[i + nx * kf] === 1 || m[i + nx * kf] === 4)) kf++;
      if (kf >= nz) { zs[i] = zfix[i] = -1; zcap[i] = -1; continue; }
      let kx = kf; while (kx > 0 && m[i + nx * (kx - 1)] === 4) kx--;
      let ku = kf; while (ku < nz && m[i + nx * ku] !== 1 && m[i + nx * ku] !== 4) ku++;
      zfix[i] = kx * dx; zcap[i] = ku * dx;
      // the bed height inside its cells: the drawn bed's coverage (raster.frac) when the interface gives
      // it, so a gentle slope starts as a slope and not as whole-cell steps; else the whole cells
      if (raster.frac) { let z = kx * dx; for (let k = kx; k < ku; k++) { const fr = raster.frac[i + nx * k]; if (!(fr > 0)) break; z += Math.min(1, fr) * dx; } zs[i] = Math.max(Math.min(z, ku * dx - 0.01 * dx), kx * dx); }
      else zs[i] = kf * dx;
    }
    M.bed0 = { solid: solid.slice(), ks: ks.slice(), comp, tdep, zs, zfix, zcap, mat: m.slice() };
    // plant columns: the lowest run of plant cells in each column (base, length, stem, Cd, EI)
    const pl = new Float32Array(5 * nx), pf = new Float32Array(N).fill(1); M.hasPlants = false;
    for (let i = 0; i < nx; i++) {
      let k0 = 0; while (k0 < nz && m[i + nx * k0] !== 3) k0++; if (k0 >= nz) continue;
      let k1 = k0; while (k1 < nz && m[i + nx * k1] === 3) k1++;
      const c0 = i + nx * k0; M.hasPlants = true;
      pl[5 * i] = k0 * dx; pl[5 * i + 1] = (k1 - k0) * dx; pl[5 * i + 2] = raster.p3[c0] || 0.006; pl[5 * i + 3] = raster.p2[c0] || 1;
      pl[5 * i + 4] = raster.p4 && raster.p4[c0] > 0 ? raster.p4[c0] : 1e6;
    }
    M.device.queue.writeBuffer(M.S.B.pl, 0, pl); M.device.queue.writeBuffer(M.S.B.pf, 0, pf);
    M.device.queue.writeBuffer(M.S.B.po, 0, new Float32Array(2 * nx).fill(1));
    M.plantHost = null;
    writeBed();
    M.device.queue.writeBuffer(M.S.B.dgX, 0, dgX); M.device.queue.writeBuffer(M.S.B.dgZ, 0, dgZ); M.device.queue.writeBuffer(M.S.B.ks, 0, ks);
    M.applyParams();
  };

  // bed height at a column: the solid cells counted up from the floor
  function bedAt(i) {
    const g = M.grid, s = M.solidHost; if (!s) return 0;
    let k = 0; while (k < g.nz && s[i + g.nx * k]) k++; return k * g.dx;
  }
  // (re)write the starting bed: solids, roughness, mix, deposit times, bed heights
  function writeBed() {
    const b = M.bed0, q = M.device.queue, B = M.S.B; if (!b) return;
    const nx = M.grid.nx, N = b.solid.length, bed = new Float32Array(3 * nx), cell = new Float32Array(4 * N), colx = new Float32Array(6 * nx);
    for (let i = 0; i < nx; i++) { bed[3 * i] = b.zs[i]; bed[3 * i + 1] = b.zfix[i]; bed[3 * i + 2] = b.zcap[i]; colx[6 * i + 3] = colx[6 * i + 4] = colx[6 * i + 5] = 1 / 3; }
    for (let c = 0; c < N; c++) { cell[4 * c] = b.comp[3 * c]; cell[4 * c + 1] = b.comp[3 * c + 1]; cell[4 * c + 2] = b.comp[3 * c + 2]; cell[4 * c + 3] = b.tdep[c]; }
    q.writeBuffer(B.solid, 0, b.solid); q.writeBuffer(B.ks, 0, b.ks); q.writeBuffer(B.bed, 0, bed); q.writeBuffer(B.cell, 0, cell); q.writeBuffer(B.colx, 0, colx);
    q.writeBuffer(B.cs, 0, new Float32Array(3 * N));
    M.solidHost = b.solid; M.zsHost = b.zs.slice(); M.bedRange = [0, 0];
  }
  M.applyParams = function () {
    if (!M.grid) return;
    const g = M.grid, p = M.p, L = g.nx * g.dx, river = p.type !== 'wave';
    // river: water comes in between the inlet bed and the still-water level, and leaves between
    // the outlet bed and the tailwater depth above it, so inflow and outflow carry the same q
    const zIn = bedAt(0), zOut = bedAt(g.nx - 1);
    const hin = Math.max(p.h0, zIn + g.dx), hout = river ? zOut + Math.max(p.hout, g.dx) : p.h0;
    let S = 0, om = 0, wl = 0;
    if (!river && p.H > 0 && p.T > 0) { const w = pistonStroke(p.H, p.T, p.h0 - zIn); S = w.S; om = w.om; wl = w.L; }
    if (river) om = 1 / Math.max(2, 0.5 * L / Math.sqrt(G * Math.max(p.h0 - zIn, g.dx)));   // soft start
    M.wave = { S, om, L: wl };
    const sponge = river || p.farEnd === 'wall' ? 0 : Math.min(Math.max(wl, 0.15 * L), 0.4 * L);
    const P = { nx: g.nx, nz: g.nz, dx: g.dx, dt: M.dt, g: G, mode: river ? 0 : 1,
      uin: river ? p.q / (hin - zIn) : 0, hin, uout: river ? p.q / Math.max(hout - zOut, g.dx) : 0, hout, S, om, sx0: L - sponge, srate: 40 / Math.max(p.T, 0.5),
      zin: zIn, zout: zOut, q: river ? p.q : 0, nu: NU_WATER, cs: M.turbulence === false ? 0 : CS_SMAG,
      amp: river ? 0 : 0.5 * (p.H || 0), absorb: !river && p.absorb !== false && p.absorb !== 0 ? 1 : 0 };
    M.P = P;
    if (M.ok && M.S) {
      writeP(M.S.P0, P);
      const sp = new Float32Array(16), gp = GRAIN_D.map(grainProps);
      for (let k = 0; k < 3; k++) { sp[k] = GRAIN_D[k]; sp[3 + k] = gp[k].ws; sp[6 + k] = gp[k].thc; }
      sp[9] = p.morfac || 10; sp[10] = SED.porosity; sp[11] = Math.tan(SED.repose * Math.PI / 180); sp[12] = SED.slope;
      sp[13] = p.sediment ? 1 : 0; sp[14] = river && p.feed !== false ? 1 : 0; sp[15] = SED.s - 1;   // supply: a river brings in what it carries (zero gradient) unless clear water is chosen; a paddle brings none
      M.device.queue.writeBuffer(M.S.SPU, 0, sp);
    }
  };

  // Fill still water to h0 (fractional top cell), zero velocity, time 0.
  M.gen = 0;
  M.reset = function () {
    // readbacks already in flight carry the old run: bumping the generation makes them drop their
    // result when they land, so a restart is not painted over by the bed or surface it replaced
    M.gen++;
    M.t = 0; M.etaHost = null; M.plantHost = null; M.dyeOn = false; M.zsHost = null; M.bedRange = [0, 0]; M.colHost = null;
    if (M.S) { const pz = new Float32Array(8 * (M.S.nT + M.S.nF)); for (let j = 0; j < M.S.nT + M.S.nF; j++) pz[8 * j + 6] = -1; M.device.queue.writeBuffer(M.S.B.P, 0, pz);
      const ps = new Float32Array(8 * M.S.nS); for (let j = 0; j < M.S.nS; j++) ps[8 * j + 6] = -1; M.device.queue.writeBuffer(M.S.B.PS, 0, ps); }
    M.expClear = true; clearComb(); clearWires();
    if (M.S) M.device.queue.writeBuffer(M.S.B.dye, 0, new Float32Array(M.S.N));
    if (!M.ok || !M.S || !M.grid) return;
    const { nx, nz, dx } = M.grid, N = nx * nz, f = new Float32Array(N), sol = M.solidHost || new Uint32Array(N);
    const h0 = M.p.h0;
    for (let k = 0; k < nz; k++) { const fr = Math.min(Math.max((h0 - k * dx) / dx, 0), 1); if (fr <= 0) break; for (let i = 0; i < nx; i++) f[i + nx * k] = sol[i + nx * k] ? 0 : fr; }
    writeBed();
    const B = M.S.B, q = M.device.queue;
    q.writeBuffer(B.f, 0, f); q.writeBuffer(B.u, 0, new Float32Array(M.S.NU)); q.writeBuffer(B.w, 0, new Float32Array(M.S.NW)); q.writeBuffer(B.tb, 0, new Float32Array(4));
    // everything else the last run left in the water: suspended sand, dye in transit, turbulence,
    // the solver's working fields
    { const ce = M.device.createCommandEncoder();
      for (const n of ['cs', 'cs2', 'dye2', 'nt', 'u2', 'w2', 'uvA', 'uvB', 'wvA', 'wvB', 'uM', 'wM', 'flx', 'p', 'ust', 'ustK', 'qc', 'qb']) if (B[n]) ce.clearBuffer(B[n]);
      q.submit([ce.finish()]); }
    let v = 0; for (let c = 0; c < N; c++) v += f[c]; M.vol0 = v * dx * dx; M.vol = M.vol0; M.vmax = 0;
    M.applyParams();
    const e = M.device.createCommandEncoder(), pass = e.beginComputePass(); run(pass, M.S.d.surface); pass.end(); q.submit([e.finish()]);
  };

  function clearMG(enc) { for (const v of M.S.L) enc.clearBuffer(v.x); }
  function vcycle(pass) {
    const L = M.S.L, top = L.length - 1;
    for (let l = 0; l < top; l++) { run(pass, L[l].red); run(pass, L[l].black); run(pass, L[l].red); run(pass, L[l].black); run(pass, L[l].res); run(pass, L[l].restr); }
    for (let j = 0; j < 10; j++) { run(pass, L[top].red); run(pass, L[top].black); }
    for (let j = 0; j < 10; j++) { run(pass, L[top].black); run(pass, L[top].red); }
    for (let l = top - 1; l >= 0; l--) { run(pass, L[l].prol); run(pass, L[l].black); run(pass, L[l].red); run(pass, L[l].black); run(pass, L[l].red); }
  }
  function encodePressure(enc) {
    const d = M.S.d;
    let pass = enc.beginComputePass(); run(pass, d.rhs); for (const v of M.S.L.slice(0, -1)) run(pass, v.coarsen); pass.end();
    clearMG(enc);
    pass = enc.beginComputePass(); vcycle(pass); run(pass, d.dotZR); run(pass, d.fin0); pass.end();
    enc.copyBufferToBuffer(M.S.B.z, 0, M.S.B.s, 0, M.S.N * 4);
    for (let it = 0; it < (M.kcg || K_CG); it++) {
      pass = enc.beginComputePass(); run(pass, d.applyA); run(pass, d.dotSQ); run(pass, d.finSQ); run(pass, d.alpha); run(pass, d.updPR); pass.end();
      clearMG(enc);
      pass = enc.beginComputePass(); vcycle(pass); run(pass, d.dotZR); run(pass, d.fin4); run(pass, d.beta); run(pass, d.updS); pass.end();
    }
  }
  function encodeStep(enc, n) {
    const d = M.S.d, B = M.S.B;
    let pass = enc.beginComputePass();
    run(pass, d.surface); run(pass, d.classify); run(pass, d.validU); run(pass, d.validW);
    run(pass, d.exUab); run(pass, d.exUba); run(pass, d.exUab); run(pass, d.exWab); run(pass, d.exWba); run(pass, d.exWab);
    pass.end();
    enc.copyBufferToBuffer(B.u2, 0, B.u, 0, M.S.NU * 4); enc.copyBufferToBuffer(B.w2, 0, B.w, 0, M.S.NW * 4);
    pass = enc.beginComputePass(); run(pass, d.bcU); run(pass, d.bcW);
    if (M.advection === 'sl') { run(pass, d.advU); run(pass, d.advW); pass.end(); }
    else if (M.maccormack === false) { run(pass, d.advFU); run(pass, d.advFW); pass.end(); }
    else {
      run(pass, d.advFU); run(pass, d.advFW); run(pass, d.macU); run(pass, d.macW); pass.end();
      enc.copyBufferToBuffer(B.uM, 0, B.u2, 0, M.S.NU * 4); enc.copyBufferToBuffer(B.wM, 0, B.w2, 0, M.S.NW * 4);
    }
    enc.copyBufferToBuffer(B.u2, 0, B.u, 0, M.S.NU * 4); enc.copyBufferToBuffer(B.w2, 0, B.w, 0, M.S.NW * 4);
    pass = enc.beginComputePass(); run(pass, d.spongeU); run(pass, d.spongeW); run(pass, d.bcU); run(pass, d.bcW); run(pass, d.nut); run(pass, d.diffU); run(pass, d.diffW); pass.end();
    enc.copyBufferToBuffer(B.u2, 0, B.u, 0, M.S.NU * 4); enc.copyBufferToBuffer(B.w2, 0, B.w, 0, M.S.NW * 4);
    pass = enc.beginComputePass(); if (M.hasPlants) run(pass, d.plantBend); run(pass, d.dragU); run(pass, d.dragW); run(pass, d.bcU); run(pass, d.bcW); run(pass, d.classify); pass.end();
    encodePressure(enc);
    pass = enc.beginComputePass();
    run(pass, d.projU); run(pass, d.projW); run(pass, d.bcU); run(pass, d.bcW);
    run(pass, d.validU); run(pass, d.validW); run(pass, d.exUab); run(pass, d.exUba); run(pass, d.exWab); run(pass, d.exWba);
    run(pass, d.bcU); run(pass, d.bcW);
    pass.end();
    enc.copyBufferToBuffer(B.f, 0, B.f0, 0, M.S.N * 4);
    pass = enc.beginComputePass();
    if (n & 1) { run(pass, d.fluxX); run(pass, d.updX); run(pass, d.fluxZ); run(pass, d.updZ); }
    else { run(pass, d.fluxZ); run(pass, d.updZ); run(pass, d.fluxX); run(pass, d.updX); }
    run(pass, d.tick);
    if (M.dyeOn) { run(pass, d.classify); run(pass, d.dyeAdv); pass.end(); enc.copyBufferToBuffer(B.dye2, 0, B.dye, 0, M.S.N * 4); pass = enc.beginComputePass(); }
    if (M.p.sediment) {
      run(pass, d.surface); run(pass, d.classify); run(pass, d.sedShear); run(pass, d.susAdv);
      pass.end(); enc.copyBufferToBuffer(B.cs2, 0, B.cs, 0, M.S.N * 12); pass = enc.beginComputePass();
      run(pass, d.susBed); run(pass, d.sedFlux); run(pass, d.sedExner);
      for (let a = 0; a < 4; a++) { run(pass, d.sedAvalR); run(pass, d.sedAvalB); }
      run(pass, d.sedCells);
    }
    pass.end();
  }
  function encodeDiag(enc) {
    const d = M.S.d, pass = enc.beginComputePass();
    run(pass, d.surface); run(pass, d.classify); run(pass, d.speed); run(pass, d.redMax); run(pass, d.finMax); run(pass, d.redVol); run(pass, d.finVol);
    pass.end();
  }

  // Time step from the last measured top speed (one frame late, which is fine at CFL 0.4).
  function chooseDt() {
    const g = M.grid, c = Math.sqrt(G * Math.max(M.p.h0, g.dx));
    const vin = M.P ? Math.max(M.P.uin, M.P.uout, 0.5 * M.P.S * M.P.om) : 0;
    const vref = Math.max(M.vmax, vin, 0.3 * c, 0.05);
    return Math.min(CFL * g.dx / vref, 0.05);
  }

  // Advance the flume by about dtWall x speed of model time, at most maxSteps steps.
  M.frame = function (dtWall, running) {
    if (!M.ok || !M.S) return;
    M.frameNo++;
    if (running && !M.busy) {
      M.dt = chooseDt(); M.applyParams();
      const want = Math.min(dtWall, 0.05) * (M.p.speed || 1);
      // never queue more than about 40 ms of GPU work in one frame (a slow GPU must not trip
      // the browser's watchdog); the first frame does one step to measure
      const budget = M.msStep > 0 ? Math.max(1, Math.floor(40 / M.msStep)) : 1;
      const n = Math.max(1, Math.min(M.maxSteps || 10, budget, Math.ceil(want / M.dt)));
      const enc = M.device.createCommandEncoder();
      for (let j = 0; j < n; j++) encodeStep(enc, M.stepCount = (M.stepCount || 0) + 1);
      encodeDiag(enc);
      M.device.queue.submit([enc.finish()]);
      M.t += n * M.dt; M.stepsFrame = n; M.lastAdvance = n * M.dt;
      M.busy = true; const t0 = performance.now();
      M.device.queue.onSubmittedWorkDone().then(() => { M.msStep = (performance.now() - t0) / n; M.busy = false; }, () => { M.busy = false; });
    }
    if (M.frameNo % 6 === 0 && !M.readBusy) readScalars();
    if (M.wantSection && M.frameNo % 10 === 3 && !M.colBusy) readColumns();
    // particles once a frame, by the model time this frame advanced (0 when paused)
    M.stepParticles(running && M.lastAdvance ? M.lastAdvance : 0);
    M.lastAdvance = 0;
    if (M.watching && M.frameNo % 2 === 0 && !M.etaBusy) readSurface();
    if (M.p.sediment && running && M.frameNo % 30 === 0 && !M.bedBusy) readBed();
    if (M.hasPlants && running && M.frameNo % 3 === 0 && !M.plantBusy) readPlants();
  };
  // the moving bed, for drawing: which cells are sediment, their mix and deposit time
  function readBed() {
    if (!M.S || !M.bed0) return;
    M.bedBusy = true; const gen = M.gen;
    const B = M.S.B, names = ['solid', 'cell', 'bed'], st = {}, e = M.device.createCommandEncoder();
    for (const n of names) { st[n] = M.device.createBuffer({ size: B[n].size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); e.copyBufferToBuffer(B[n], 0, st[n], 0, B[n].size); }
    M.device.queue.submit([e.finish()]);
    Promise.all(names.map(n => st[n].mapAsync(GPUMapMode.READ))).then(() => {
      if (gen !== M.gen) { for (const n of names) { st[n].unmap(); st[n].destroy(); } M.bedBusy = false; return; }
      const N = M.grid.nx * M.grid.nz, solid = new Uint32Array(st.solid.getMappedRange().slice(0, N * 4));
      const cell = new Float32Array(st.cell.getMappedRange().slice(0, 4 * N * 4)), bed = new Float32Array(st.bed.getMappedRange().slice(0, 3 * M.grid.nx * 4));
      const comp = new Float32Array(3 * N), tdep = new Float32Array(N), zs = new Float32Array(M.grid.nx);
      for (let c = 0; c < N; c++) { comp[3 * c] = cell[4 * c]; comp[3 * c + 1] = cell[4 * c + 1]; comp[3 * c + 2] = cell[4 * c + 2]; tdep[c] = cell[4 * c + 3]; }
      for (let i = 0; i < zs.length; i++) zs[i] = bed[3 * i];
      for (const n of names) { st[n].unmap(); st[n].destroy(); }
      const m0 = M.bed0.mat, mat = new Uint8Array(N);
      for (let c = 0; c < N; c++) mat[c] = m0[c] === 1 ? 1 : solid[c] ? 4 : (m0[c] === 4 ? 0 : m0[c]);
      let lo = 0, hi = 0; for (let i = 0; i < zs.length; i++) if (M.bed0.zs[i] >= 0) { const d = zs[i] - M.bed0.zs[i]; lo = Math.min(lo, d); hi = Math.max(hi, d); }
      M.zsHost = zs; M.bedRange = [lo, hi];
      if (M.api && M.api.bedChanged) M.api.bedChanged({ mat, comp, tdep, zs, zfix: M.bed0.zfix });
      M.bedBusy = false;
    }, () => { M.bedBusy = false; });
  }
  // the smoke comb: 14 release points, rings of 600 (10 s of filament at one emission per 1/60 s of
  // model time); every 10th particle is a pulse mark (Marey's chronograph), so the spacing of the
  // marks along a filament is its speed
  M.comb = { on: false, x: null, n: -1, count: 0, acc: 0 };
  M.setComb = function (x) { M.comb.x = x; clearComb(); };
  function clearComb() {
    M.comb.n = -1; M.comb.count = 0; M.comb.acc = 0;
    if (M.S) { const z = new Float32Array(4 * COMB.S * COMB.M); for (let j = 0; j < COMB.S * COMB.M; j++) z[4 * j + 2] = -1; M.device.queue.writeBuffer(M.S.B.CB, 0, z); }
  }
  M.combX = function () {
    if (M.comb.x != null) return M.comb.x;
    return M.grid ? (M.p.type === 'wave' ? 0.3 : 0.04) * M.grid.nx * M.grid.dx : null;
  };
  function stepComb(dtp) {
    if (!M.comb.on || !(dtp > 0)) return;
    const C = M.comb; C.acc += dtp;
    const k = Math.min(8, Math.floor(C.acc / COMB.te)); C.acc -= k * COMB.te;
    const xs = M.combX(), parts = Math.max(k, 1);
    for (let j = 0; j < parts; j++) {
      const emit = j < k ? 1 : 0;
      if (emit) { C.n = (C.n + 1) % COMB.M; C.count++; }
      M.device.queue.writeBuffer(M.S.CU, 0, new Float32Array([dtp / parts, emit, Math.max(C.n, 0), xs, COMB.S, COMB.M, C.count, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
      const e = M.device.createCommandEncoder(), ps = e.beginComputePass(); run(ps, M.S.d.combStep); ps.end(); M.device.queue.submit([e.finish()]);
    }
  }
  // bubble wires: up to four, placed by the interface; a release every Tr seconds, Tr chosen so a
  // timeline moves about a quarter depth between releases (waves: an eighth of a period)
  M.wires = { on: false, xs: [], n: -1, count: 0, acc: 0 };
  M.setWires = function (xs) { M.wires.xs = xs.slice(0, WIRE.W); clearWires(); };
  M.wireXs = function () {
    if (M.wires.xs.length) return M.wires.xs;
    return M.grid ? [(M.p.type === 'wave' ? 0.3 : 0.25) * M.grid.nx * M.grid.dx] : [];
  };
  M.wireInterval = function () {
    const p = M.p, h = Math.max(p.h0 - (M.P ? M.P.zin : 0), M.grid ? 4 * M.grid.dx : 0.05);
    if (p.type === 'wave') return Math.max(0.05, (p.T || 2) / 8);
    return Math.min(3, Math.max(0.08, 0.25 * h / Math.max(p.q / h, 0.02)));
  };
  function clearWires() {
    M.wires.n = -1; M.wires.count = 0; M.wires.acc = 1e9;   // release at once on the next step
    if (M.S) { const z = new Float32Array(4 * WIRE.W * WIRE.R * WIRE.K); for (let j = 0; j < WIRE.W * WIRE.R * WIRE.K; j++) z[4 * j + 2] = -1; M.device.queue.writeBuffer(M.S.B.WB, 0, z); }
  }
  function stepWires(dtp) {
    if (!M.wires.on || !(dtp > 0)) return;
    const Wr = M.wires, xs = M.wireXs(), Tr = M.wireInterval();
    Wr.acc = Math.min(Wr.acc + dtp, Tr + dtp);
    const rel = Wr.acc >= Tr ? 1 : 0; if (rel) { Wr.acc = 0; Wr.n = (Wr.n + 1) % WIRE.R; Wr.count++; }
    const u = new Float32Array(16); u.set([dtp, rel, Math.max(Wr.n, 0), 0, xs.length, WIRE.R, WIRE.K, 0]); for (let j = 0; j < xs.length; j++) u[8 + j] = xs[j];
    M.device.queue.writeBuffer(M.S.WU, 0, u);
    const e = M.device.createCommandEncoder(), ps = e.beginComputePass(); run(ps, M.S.d.wireStep); ps.end(); M.device.queue.submit([e.finish()]);
  }
  M.stepParticles = function (dtp) {
    if (!M.ok || !M.S || M.particles === false) return;
    M.pFrame = (M.pFrame || 0) + 1;
    const hh = Math.max(M.p.h0 - (M.P ? M.P.zin : 0), M.grid.dx);
    M.device.queue.writeBuffer(M.S.PU, 0, new Float32Array([dtp, M.t, M.pFrame, M.S.nT, M.S.nF, hh, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
    const e = M.device.createCommandEncoder(), ps = e.beginComputePass(); run(ps, M.S.d.partStep);
    if (M.streak) { M.device.queue.writeBuffer(M.S.PU2, 0, new Float32Array([dtp, M.t, M.pFrame, M.S.nS, 0, hh, 7919, 0, 0, 0, 0, 0, 0, 0, 0, 0])); run(ps, M.S.d.partStepS); }
    ps.end(); M.device.queue.submit([e.finish()]);
    M.pAdvance = (M.pAdvance || 0) + dtp;          // model time the exposure has to take in at the next draw
    stepComb(dtp); stepWires(dtp);
  };
  // dye tool: release dye in a disc at (x, z) of radius r (m)
  M.injectDye = function (x, z, r) {
    if (!M.ok || !M.S) return;
    M.device.queue.writeBuffer(M.S.DIU, 0, new Float32Array([x, z, r, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
    const e = M.device.createCommandEncoder(), pass = e.beginComputePass(); run(pass, M.S.d.dyeInject); pass.end(); M.device.queue.submit([e.finish()]);
    M.dyeOn = true;
  };
  M.clearDye = function () { if (M.S) M.device.queue.writeBuffer(M.S.B.dye, 0, new Float32Array(M.S.N)); M.dyeOn = false; };
  // plants' bend per column, for drawing the stems
  M.plantSample = function () { return M.hasPlants ? M.plantHost : null; };
  function readPlants() {
    if (!M.S) return;
    M.plantBusy = true; const B = M.S.B.po, n = M.grid.nx, gen = M.gen;
    const st = M.device.createBuffer({ size: B.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = M.device.createCommandEncoder(); e.copyBufferToBuffer(B, 0, st, 0, B.size); M.device.queue.submit([e.finish()]);
    st.mapAsync(GPUMapMode.READ).then(() => {
      const a = new Float32Array(st.getMappedRange().slice(0, 2 * n * 4)); st.unmap(); st.destroy();
      if (gen !== M.gen) { M.plantBusy = false; return; }
      const le = new Float32Array(n), dir = new Float32Array(n); for (let i = 0; i < n; i++) { le[i] = a[2 * i]; dir[i] = a[2 * i + 1]; }
      M.plantHost = { le, dir }; M.plantBusy = false;
    }, () => { M.plantBusy = false; });
  }
  // gauges: the surface height per column, read back every other frame while gauges exist
  M.watchSurface = function (on) { M.watching = !!on; };
  M.surfaceSample = function () { return M.etaHost || null; };
  M.gaugeWindow = function () { return M.p.type === 'wave' ? Math.max(3 * (M.p.T || 2), 6) : 10; };
  function readSurface() {
    if (!M.S) return;
    M.etaBusy = true; const t = M.t, n = M.grid.nx, B = M.S.B.eta, gen = M.gen;
    const st = M.device.createBuffer({ size: B.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = M.device.createCommandEncoder(); e.copyBufferToBuffer(B, 0, st, 0, B.size); M.device.queue.submit([e.finish()]);
    st.mapAsync(GPUMapMode.READ).then(() => {
      if (gen === M.gen) M.etaHost = { t, eta: new Float32Array(st.getMappedRange().slice(0, n * 4)), dx: M.grid.dx };
      st.unmap(); st.destroy(); M.etaBusy = false;
    }, () => { M.etaBusy = false; });
  }
  // the measured section's columns: q, h, eta, smoothed in model time (tau a few seconds, so the
  // energy line shows the steady state, not every eddy)
  function readColumns() {
    if (!M.S) return;
    M.colBusy = true; const gen = M.gen, t = M.t, n = M.grid.nx, B = M.S.B;
    const st = M.device.createBuffer({ size: B.egl.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = M.device.createCommandEncoder(), ps = e.beginComputePass(); run(ps, M.S.d.colQ); ps.end();
    e.copyBufferToBuffer(B.egl, 0, st, 0, B.egl.size); M.device.queue.submit([e.finish()]);
    st.mapAsync(GPUMapMode.READ).then(() => {
      const a = new Float32Array(st.getMappedRange().slice(0, 3 * n * 4)); st.unmap(); st.destroy(); M.colBusy = false;
      if (gen !== M.gen) return;
      const H = M.colHost;
      const wetNow = (i) => a[3 * i + 1] > 2 * M.grid.dx && a[3 * i + 2] > 0;
      if (!H || H.n !== n) { M.colHost = { n, t, q: new Float32Array(n), h: new Float32Array(n), eta: new Float32Array(n), wet: new Float32Array(n), w: 0 }; for (let i = 0; i < n; i++) { M.colHost.q[i] = a[3 * i]; M.colHost.h[i] = a[3 * i + 1]; M.colHost.eta[i] = a[3 * i + 2]; M.colHost.wet[i] = wetNow(i) ? 1 : 0; } return; }
      const tau = M.p.type === 'wave' ? 3 * Math.max(M.p.T || 2, 0.5) : 3.0, al = 1 - Math.exp(-Math.max(t - H.t, 0) / tau);
      if (al <= 0) return;
      for (let i = 0; i < n; i++) {
        const wt = wetNow(i); H.wet[i] += al * ((wt ? 1 : 0) - H.wet[i]);
        if (!wt) continue;                       // a dry moment says nothing about the mean surface
        if (H.eta[i] <= 0) { H.q[i] = a[3 * i]; H.h[i] = a[3 * i + 1]; H.eta[i] = a[3 * i + 2]; continue; }
        H.q[i] += al * (a[3 * i] - H.q[i]); H.h[i] += al * (a[3 * i + 1] - H.h[i]); H.eta[i] += al * (a[3 * i + 2] - H.eta[i]);
      }
      H.t = t; H.w = Math.min(1, H.w + al);
    }, () => { M.colBusy = false; });
  }
  // Lines for the measured section, in metres: named levels (horizontal), lines along the flume,
  // and marks (energy losses). River: tailwater, the energy line E = eta + U^2/2g, and the critical
  // depth over the bed; wave: still water, nominal crest and trough, and the mean water level.
  M.sectionLines = function () {
    if (!M.ok || !M.grid) return null;
    const g = M.grid, p = M.p, river = p.type !== 'wave', out = { levels: [], lines: [], marks: [] };
    if (river) out.levels.push({ z: bedAt(g.nx - 1) + Math.max(p.hout, g.dx), label: 'tailwater', style: 'tail' });
    else {
      out.levels.push({ z: p.h0, label: 'still water', style: 'still' });
      if (p.H > 0) { out.levels.push({ z: p.h0 + p.H / 2, label: 'wave crest', style: 'faint' }); out.levels.push({ z: p.h0 - p.H / 2, label: 'wave trough', style: 'faint' }); }
    }
    const H = M.colHost; if (!H || H.w < 0.3) return out;
    const n = g.nx, dx = g.dx, ok = (i) => H.wet[i] > 0.9 && H.h[i] > 2 * dx && H.eta[i] > 0;
    // smooth over a few cells
    const sm = (a, r) => { const o = new Float32Array(n); for (let i = 0; i < n; i++) { let s = 0, c = 0; for (let j = Math.max(0, i - r); j <= Math.min(n - 1, i + r); j++) if (ok(j)) { s += a[j]; c++; } o[i] = c ? s / c : NaN; } return o; };
    const r = Math.max(2, Math.round(0.02 * n / 20));
    const eta = sm(H.eta, r), h = sm(H.h, r), q = sm(H.q, r);
    const poly = (f) => { const pts = []; let seg = []; for (let i = 0; i < n; i++) { const z = ok(i) ? f(i) : NaN; if (isFinite(z)) seg.push([(i + 0.5) * dx, z]); else if (seg.length) { if (seg.length > 2) pts.push(seg); seg = []; } } if (seg.length > 2) pts.push(seg); return pts; };
    if (river) {
      const E = new Float32Array(n); for (let i = 0; i < n; i++) { const U = q[i] / Math.max(h[i], dx); E[i] = eta[i] + U * U / (2 * G); }
      out.lines.push({ segs: poly((i) => E[i]), label: 'energy', style: 'energy' });
      const yc = Math.cbrt(Math.max(p.q, 0) ** 2 / G);
      if (yc > dx) out.lines.push({ segs: poly((i) => eta[i] - h[i] + yc), label: 'critical depth', style: 'critical' });
      // energy losses: across each stretch of rapidly varied flow (where the time-mean surface slopes
      // more than 2%: a jump, a drawdown over a crest, a plunge behind a block), the energy line's
      // drop from just before to just after it; smooth stretches (friction only) are not marked
      let hm = 0, cnt = 0; for (let i = 0; i < n; i++) if (ok(i)) { hm += h[i]; cnt++; } hm = cnt ? hm / cnt : 0;
      const steep = new Uint8Array(n);
      // slope over about half a depth (small ripples do not count); the inlet and outlet zones,
      // where the boundaries shape the surface, are left out
      let hx = 0; for (let i = 0; i < n; i++) if (ok(i)) hx = Math.max(hx, h[i]);
      const sw = Math.max(2, Math.round(0.5 * hm / dx)), e0 = Math.max(sw, Math.round(3 * hx / dx));
      for (let i = e0; i < n - e0; i++) if (ok(i - sw) && ok(i + sw) && Math.abs(eta[i + sw] - eta[i - sw]) / (2 * sw * dx) > 0.02) steep[i] = 1;
      const regs = []; let i0 = -1;
      for (let i = 0; i <= n; i++) { if (i < n && steep[i]) { if (i0 < 0) i0 = i; } else if (i0 >= 0) { regs.push([i0, i - 1]); i0 = -1; } }
      const gap = Math.max(2, Math.round(2 * hx / dx)), merged = [];     // the deep water's depth sets the scale (a long shallow jet must not)
      for (const r of regs) { const last = merged[merged.length - 1]; if (last && r[0] - last[1] <= gap) last[1] = r[1]; else merged.push(r.slice()); }
      const mg = Math.max(2, Math.round(0.5 * hm / dx)), thr = Math.max(0.04 * hm, 1e-3);
      const found = [];
      for (const [r0, r1] of merged) {
        let ia = Math.max(0, r0 - mg), ib = Math.min(n - 1, r1 + mg);
        while (ia < r0 && !isFinite(E[ia])) ia++; while (ib > r1 && !isFinite(E[ib])) ib--;
        if (!isFinite(E[ia]) || !isFinite(E[ib])) continue;
        const d = E[ia] - E[ib]; if (d <= thr) continue;
        // draw it where 90% of the loss happens (the ends of a long stretch are often friction only)
        let ja = ia, jb = ib;
        while (ja < ib && !(isFinite(E[ja]) && E[ia] - E[ja] >= 0.05 * d)) ja++;
        while (jb > ja && !(isFinite(E[jb]) && E[jb] - E[ib] >= 0.05 * d)) jb--;
        found.push({ ia: Math.max(ia, ja - 1), ib: Math.min(ib, jb + 1), d, Ea: E[ia], Eb: E[ib] });
      }
      found.sort((p1, p2) => p2.d - p1.d);
      for (const f of found.slice(0, 4)) out.marks.push({ x0: (f.ia + 0.5) * dx, x1: (f.ib + 0.5) * dx, z0: f.Ea, z1: f.Eb, label: 'ΔE ' + fmtM(f.d) });
    } else {
      out.lines.push({ segs: poly((i) => eta[i]), label: 'mean water level', style: 'mean' });
    }
    return out;
  };
  function fmtM(m) { return m >= 1 ? m.toFixed(2) + ' m' : m >= 0.01 ? (m * 100).toFixed(1) + ' cm' : (m * 1000).toFixed(1) + ' mm'; }
  function readScalars() {
    M.readBusy = true; const gen = M.gen;
    const st = M.device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = M.device.createCommandEncoder(); e.copyBufferToBuffer(M.S.B.scal, 0, st, 0, 32); M.device.queue.submit([e.finish()]);
    st.mapAsync(GPUMapMode.READ).then(() => {
      const a = new Float32Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy();
      if (gen !== M.gen) { M.readBusy = false; return; }
      if (isFinite(a[7])) M.vmax = a[7];
      if (isFinite(a[6]) && a[6] > 0) M.vol = a[6] * M.grid.dx * M.grid.dx;
      M.readBusy = false;
    }, () => { M.readBusy = false; });
  }

  M.render = function (view) {
    if (!M.ok || !M.S) return;
    if (M.readback && !view._rb) return renderReadback(view);
    if (!M.ctx && !view.target) return;
    if (!M.render_) {
      const module = M.device.createShaderModule({ code: RENDER });
      const p = M.device.createRenderPipeline({ layout: 'auto', vertex: { module, entryPoint: 'vs' },
        fragment: { module, entryPoint: 'fs', targets: [{ format: M.format }] }, primitive: { topology: 'triangle-list' } });
      const U = M.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const B = M.S.B;
      const bg = M.device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [U, B.f, B.solid, B.u, B.w, B.eta, B.dye, B.cs].map((b, j) => ({ binding: j, resource: { buffer: b } })) });
      const mp = M.device.createShaderModule({ code: RENDER_P });
      const pp = M.device.createRenderPipeline({ layout: 'auto', vertex: { module: mp, entryPoint: 'vs' },
        fragment: { module: mp, entryPoint: 'fs', targets: [{ format: M.format, blend: { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } } }] },
        primitive: { topology: 'triangle-list' } });
      const pbg = M.device.createBindGroup({ layout: pp.getBindGroupLayout(0), entries: [U, B.P].map((b, j) => ({ binding: j, resource: { buffer: b } })) });
      M.render_ = { p, U, bg, pp, pbg };
    }
    const g = M.grid, r = view.dpr || 1;
    const v = new Float32Array([view.x0, view.z0, view.s * r, view.ve, view.cw * r, view.ch * r, g.dx, g.nx * g.dx, g.nx, g.nz, M.colorMode,
      g.nz * g.dx, M.t, Math.max(M.vmax, 0.05), Math.max(M.p.h0, g.dx), r]);
    M.device.queue.writeBuffer(M.render_.U, 0, v);
    const enc = M.device.createCommandEncoder();
    const tgt = view.target || M.ctx.getCurrentTexture();
    const exp = M.streak ? exposurePass(enc, view, tgt.width, tgt.height) : null;
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: tgt.createView(), loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }] });
    pass.setPipeline(M.render_.p); pass.setBindGroup(0, M.render_.bg); pass.draw(3);
    if (exp) { pass.setPipeline(exp.p); pass.setBindGroup(0, exp.bg); pass.draw(3); }
    if (M.comb.on) {
      if (!M.render_.cp || M.render_.cpBuf !== M.S.B.CB) {
        const cm = M.device.createShaderModule({ code: COMB_LINE });
        const cp = M.render_.cp || M.device.createRenderPipeline({ layout: 'auto', vertex: { module: cm, entryPoint: 'vs' },
          fragment: { module: cm, entryPoint: 'fs', targets: [{ format: M.format, blend: { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } } }] }, primitive: { topology: 'triangle-list' } });
        const cU = M.render_.cU || M.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        M.render_.cp = cp; M.render_.cU = cU; M.render_.cpBuf = M.S.B.CB;
        M.render_.cbg = M.device.createBindGroup({ layout: cp.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: M.render_.U } }, { binding: 1, resource: { buffer: M.S.B.CB } }, { binding: 2, resource: { buffer: cU } }] });
      }
      M.device.queue.writeBuffer(M.render_.cU, 0, new Float32Array([Math.max(M.comb.n, 0), COMB.M, COMB.P, 0]));
      pass.setPipeline(M.render_.cp); pass.setBindGroup(0, M.render_.cbg); pass.draw(6, COMB.S * COMB.M);
    }
    if (M.wires.on) {
      if (!M.render_.wp || M.render_.wpBuf !== M.S.B.WB) {
        const wm = M.device.createShaderModule({ code: WIRE_LINE });
        const wp = M.render_.wp || M.device.createRenderPipeline({ layout: 'auto', vertex: { module: wm, entryPoint: 'vs' },
          fragment: { module: wm, entryPoint: 'fs', targets: [{ format: M.format, blend: { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } } }] }, primitive: { topology: 'triangle-list' } });
        const wU = M.render_.wU || M.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        M.render_.wp = wp; M.render_.wU = wU; M.render_.wpBuf = M.S.B.WB;
        M.render_.wbg = M.device.createBindGroup({ layout: wp.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: M.render_.U } }, { binding: 1, resource: { buffer: M.S.B.WB } }, { binding: 2, resource: { buffer: wU } }] });
      }
      M.device.queue.writeBuffer(M.render_.wU, 0, new Float32Array([Math.max(M.wires.n, 0), WIRE.R, WIRE.K, 0]));
      pass.setPipeline(M.render_.wp); pass.setBindGroup(0, M.render_.wbg); pass.draw(6, M.wireXs().length * WIRE.R * WIRE.K);
    }
    if (M.particles !== false) {
      // with the exposure on, its flakes replace the tracers; foam stays
      const k = exp ? 'foam' : M.drawKinds;
      pass.setPipeline(M.render_.pp); pass.setBindGroup(0, M.render_.pbg); pass.draw(6, k === 'tracers' ? M.S.nT : k === 'foam' ? M.S.nF : M.S.nT + M.S.nF, 0, k === 'foam' ? M.S.nT : 0);
    }
    pass.end();
    M.device.queue.submit([enc.finish()]);
  };
  // the exposure image: two float textures at the target's size, ping-ponged; cleared when the view
  // changes (it is drawn in screen space) or the flume restarts
  let EX = null;
  M.exposure = 1.0;
  function exposurePass(enc, view, W, H) {
    const dev = M.device, fmt = 'rgba16float';
    if (!EX || EX.W !== W || EX.H !== H || EX.fmt !== M.format) {
      if (EX) for (const t of EX.tex) t.destroy();
      const mk = () => dev.createTexture({ size: [W, H], format: fmt, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
      const U = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const fm = dev.createShaderModule({ code: EXP_FADE }), lm = dev.createShaderModule({ code: EXP_LINE }), sm = dev.createShaderModule({ code: EXP_SHOW });
      const add = { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } };
      const fade = dev.createRenderPipeline({ layout: 'auto', vertex: { module: fm, entryPoint: 'vs' }, fragment: { module: fm, entryPoint: 'fs', targets: [{ format: fmt }] }, primitive: { topology: 'triangle-list' } });
      const line = dev.createRenderPipeline({ layout: 'auto', vertex: { module: lm, entryPoint: 'vs' }, fragment: { module: lm, entryPoint: 'fs', targets: [{ format: fmt, blend: add }] }, primitive: { topology: 'triangle-list' } });
      const show = dev.createRenderPipeline({ layout: 'auto', vertex: { module: sm, entryPoint: 'vs' }, fragment: { module: sm, entryPoint: 'fs', targets: [{ format: M.format, blend: add }] }, primitive: { topology: 'triangle-list' } });
      const tex = [mk(), mk()];
      const fadeBG = tex.map((t) => dev.createBindGroup({ layout: fade.getBindGroupLayout(0), entries: [{ binding: 0, resource: t.createView() }, { binding: 1, resource: { buffer: U } }] }));
      const showBG = tex.map((t) => dev.createBindGroup({ layout: show.getBindGroupLayout(0), entries: [{ binding: 0, resource: t.createView() }, { binding: 1, resource: { buffer: U } }] }));
      const lineBG = dev.createBindGroup({ layout: line.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: M.render_.U } }, { binding: 1, resource: { buffer: M.S.B.PS } }, { binding: 2, resource: { buffer: U } }] });
      EX = { W, H, fmt: M.format, tex, U, fade, line, show, fadeBG, showBG, lineBG, cur: 0, key: '', buf: M.S.B.PS };
      M.expClear = true;
    }
    if (EX.buf !== M.S.B.PS) { EX.lineBG = dev.createBindGroup({ layout: EX.line.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: M.render_.U } }, { binding: 1, resource: { buffer: M.S.B.PS } }, { binding: 2, resource: { buffer: EX.U } }] }); EX.buf = M.S.B.PS; M.expClear = true; }
    const key = [view.x0, view.z0, view.s, view.ve, view.cw, view.ch, view.dpr].join(',');
    if (key !== EX.key) { EX.key = key; M.expClear = true; }
    const T = Math.max(M.exposure || 1, 0.02), adv = M.pAdvance || 0; M.pAdvance = 0;
    // light is added per model second (a particle crossing a pixel slowly exposes it more, as on film);
    // the exposure time sets how long light stays, and the tone curve keeps long exposures from burning out
    const decay = M.expClear ? 0 : Math.exp(-adv / T), k = (M.expLight || 40) * adv, gain = (M.expGain || 2.0) * 0.12 / (0.07 + T);    // stop down as the exposure lengthens
    M.expClear = false;
    dev.queue.writeBuffer(EX.U, 0, new Float32Array([decay, k, gain, 0]));
    const src = EX.cur, dst = 1 - src; EX.cur = dst;
    const p = enc.beginRenderPass({ colorAttachments: [{ view: EX.tex[dst].createView(), loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }] });
    p.setPipeline(EX.fade); p.setBindGroup(0, EX.fadeBG[src]); p.draw(3);
    p.setPipeline(EX.line); p.setBindGroup(0, EX.lineBG); p.draw(6, M.S.nS);
    p.end();
    return { p: EX.show, bg: EX.showBG[dst] };
  }
  let rb = null;
  function renderReadback(view) {
    if (!M.rbCtx || (rb && rb.busy)) return;
    const r = view.dpr || 1, W = Math.round(view.cw * r), H = Math.round(view.ch * r);
    if (!rb || rb.W !== W || rb.H !== H) {
      if (rb) { rb.tex.destroy(); rb.buf.destroy(); }
      const bpr = Math.ceil(W * 4 / 256) * 256;
      rb = { W, H, bpr, busy: false,
        tex: M.device.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC }),
        buf: M.device.createBuffer({ size: bpr * H, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }) };
    }
    M.render(Object.assign({}, view, { target: rb.tex, _rb: true }));
    const enc = M.device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: rb.tex }, { buffer: rb.buf, bytesPerRow: rb.bpr }, [W, H]);
    M.device.queue.submit([enc.finish()]);
    rb.busy = true; const me = rb;
    me.buf.mapAsync(GPUMapMode.READ).then(() => {
      const src = new Uint8Array(me.buf.getMappedRange()), img = M.rbCtx.createImageData(W, H), d = img.data;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const o = y * me.bpr + x * 4, q = (y * W + x) * 4, a = src[o + 3];
        if (a) { const f = 255 / a; d[q] = src[o] * f; d[q + 1] = src[o + 1] * f; d[q + 2] = src[o + 2] * f; d[q + 3] = a; }
      }
      me.buf.unmap(); me.busy = false;
      M.rbCtx.putImageData(img, 0, 0);
    }, () => { me.busy = false; });
  }

  M.readouts = function () {
    if (!M.ok) return [['model', 'not running (no WebGPU)']];
    const g = M.grid, out = [
      ['time', M.t.toFixed(1) + ' s'],
      ['grid', g.nx + ' × ' + g.nz + ' (' + (g.dx * 100).toFixed(g.dx < 0.1 ? 1 : 0) + ' cm)'],
      ['top speed', M.vmax.toFixed(2) + ' m/s'],
      ['step', (M.dt * 1000).toFixed(1) + ' ms model, ' + M.msStep.toFixed(1) + ' ms GPU'],
      ['speed', (M.msStep > 0 ? Math.min(M.stepsFrame * M.dt / Math.max(M.stepsFrame * M.msStep / 1000, 1 / 60), 99) : 0).toFixed(2) + ' × real time'],
      ['water volume', (M.vol0 > 0 ? (100 * M.vol / M.vol0).toFixed(1) : '100') + ' % of start'],
    ];
    if (M.p.sediment) out.push(['bed', 'time ' + (M.t * (M.p.morfac || 10)).toFixed(0) + ' s, change ' + (M.bedRange[0] * 100).toFixed(1) + ' / +' + (M.bedRange[1] * 100).toFixed(1) + ' cm']);
    if (M.p.type === 'wave' && M.wave.L) out.push(['wavelength', M.wave.L.toFixed(2) + ' m (stroke ' + (M.wave.S * 100).toFixed(1) + ' cm)']);
    return out;
  };

  // For checks and the bench page: run n steps and wait; read a field back.
  M.debug = {
    async steps(n, dt) {
      if (dt) M.dt = dt; M.applyParams();
      for (let j = 0; j < n; j += 10) {
        const enc = M.device.createCommandEncoder();
        for (let q = 0; q < 10 && j + q < n; q++) encodeStep(enc, M.stepCount = (M.stepCount || 0) + 1);
        encodeDiag(enc); M.device.queue.submit([enc.finish()]);
        M.t += Math.min(10, n - j) * M.dt;
      }
      await M.device.queue.onSubmittedWorkDone();
    },
    write(name, data) { M.device.queue.writeBuffer(M.S.B[name], 0, data); },
    readBed() { readBed(); },                // start the page's bed readback (for the restart check)
    comb: () => COMB, wire: () => WIRE,
    async readColumns() { readColumns(); while (M.colBusy) await new Promise(r => setTimeout(r, 2)); },
    async read(name) {
      const B = M.S.B[name], n = B.size / 4;
      const st = M.device.createBuffer({ size: B.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const e = M.device.createCommandEncoder(); e.copyBufferToBuffer(B, 0, st, 0, B.size); M.device.queue.submit([e.finish()]);
      await st.mapAsync(GPUMapMode.READ); const a = new Float32Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy();
      return a.subarray(0, n);
    },
    async validateRender(width, height) {     // draw once into an offscreen texture (checks the shader compiles and binds)
      const fmt = 'rgba8unorm';
      M.format = M.format || fmt;
      const tex = M.device.createTexture({ size: [width, height], format: fmt, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      const save = M.ctx; M.ctx = { getCurrentTexture: () => tex }; M.format = fmt; M.render_ = null;
      M.device.pushErrorScope('validation');
      M.render({ x0: 0, z0: 0, s: width / (M.grid.nx * M.grid.dx), ve: 1, cw: width, ch: height, dpr: 1 });
      const err = await M.device.popErrorScope();
      await M.device.queue.onSubmittedWorkDone();
      M.ctx = save; M.render_ = null; tex.destroy();
      return err ? err.message : null;
    },
    waveNumber, pistonStroke, SRC, RENDER, grainProps, SED, GRAIN_D,
  };

  registerModel(M);
})();
