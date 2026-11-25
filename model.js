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
  const K_CG = 8;            // CG iterations per step (8 reaches the single-precision floor, see spike)
  const CFL = 0.4;
  const NU_WATER = 1e-6;     // m2/s
  const CS_SMAG = 0.17;      // Smagorinsky constant
  const POROUS_ALPHA = 1000, POROUS_BETA = 1.1;     // van Gent (1995) coefficients
  const GRAIN_D = [0.0002, 0.001, 0.01];            // m: fine sand, coarse sand, gravel (as the interface's GRAIN)

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
    binds.forEach(([n, t], j) => { decl += `@group(0) @binding(${j + 1}) var<storage, read_write> ${n}: array<${t === 'u' ? 'u32' : 'f32'}>;\n`; });
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
    dragU: K([['u', 'f'], ['solid', 'u'], ['dg', 'f'], ['ks', 'f'], ['BU', 'f']], `let W = prm.nx+1u; if (c >= W*prm.nz) {return;} let i = c % W; let k = c / W; let N = prm.nx;
      BU[c] = 1.0;
      if (i == 0u || i == prm.nx) { return; }
      let l = i-1u+N*k; let r = i+N*k; if (solid[l] != 0u || solid[r] != 0u) { return; }
      let A = 0.5*(dg[2u*l] + dg[2u*r]); var B = 0.5*(dg[2u*l+1u] + dg[2u*r+1u]);
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
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
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
  // colour by depth under the local surface
  let e = eta[u32(ci)];
  let d = max(e - z, 0.0);
  let t = clamp(d / V0.depthRef, 0.0, 1.0);
  var col = mix(vec3f(0.40, 0.76, 0.84), vec3f(0.03, 0.17, 0.34), sqrt(t));
  if (V0.mode > 0.5) {
    let W = u32(nx + 1);
    let uc = 0.5*(u[u32(ci) + W*u32(ck)] + u[u32(ci) + 1u + W*u32(ck)]);
    let wc = 0.5*(w[cc] + w[cc + u32(nx)]);
    let sp = clamp(length(vec2f(uc, wc)) / max(V0.umax, 1e-3), 0.0, 1.0);
    let c1 = mix(vec3f(0.05, 0.12, 0.35), vec3f(0.10, 0.65, 0.80), clamp(sp*2.0, 0.0, 1.0));
    col = mix(c1, vec3f(1.0, 0.86, 0.45), clamp(sp*2.0 - 1.0, 0.0, 1.0));
  }
  // bright line along the free surface
  let hl = 1.0 - clamp(abs(fv - 0.5) / (dfv*1.6), 0.0, 1.0);
  col = mix(col, vec3f(0.88, 0.97, 1.0), hl*0.85);
  let a1 = select(0.0, alpha*0.93, inside);
  return vec4f(col*a1, a1);
}
`;

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
    lab:   { type: 'river', h0: 0.40, q: 0.10, hout: 0.40, H: 0.08, T: 2.0, speed: 1, farEnd: 'beach', absorb: true },
    field: { type: 'river', h0: 6.0, q: 6.0, hout: 6.0, H: 1.5, T: 8.0, speed: 1, farEnd: 'beach', absorb: true },
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
      else if (k === 'absorb') M.p.absorb = !!s[k];
      else if (isFinite(+s[k])) M.p[k] = +s[k];
    }
    if (s.colorMode != null) M.colorMode = +s.colorMode;
    M.syncControls(); if (M.grid) M.applyParams();
  };
  M.setScale = function (scale) { M.scale = scale; M.p = Object.assign({}, DEFAULTS[scale] || DEFAULTS.lab, { type: M.p.type, farEnd: M.p.farEnd, absorb: M.p.absorb }); M.syncControls(); if (M.grid) M.applyParams(); };

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
      part: buf(Math.ceil(N / 256) + 1), scal: buf(8), tb: buf(4), eta: buf(nx),
    };
    const P0 = ubuf();
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
      speed: D('speed', P0, [B.u, B.w, B.typ, B.tmp], N),
      nut: D('nut', P0, [B.u, B.w, B.typ, B.nt], N),
      diffU: D('diffU', P0, [B.u, B.typ, B.nt, B.u2], NU), diffW: D('diffW', P0, [B.w, B.typ, B.nt, B.w2], NW),
      dragU: D('dragU', P0, [B.u, B.solid, B.dgX, B.ks, B.BU], NU), dragW: D('dragW', P0, [B.w, B.solid, B.dgZ, B.ks, B.BW], NW),
      redVol: D('red', P0, [B.f, B.part], N, { OP: 0 }, 256), finVol: D('fin', P0, [B.part, B.scal], 256, { SLOT: 6, OP: 0 }, 256),
      redMax: D('red', P0, [B.tmp, B.part], N, { OP: 1 }, 256), finMax: D('fin', P0, [B.part, B.scal], 256, { SLOT: 7, OP: 1 }, 256),
    };
    M.S = { B, L, d, P0, N, NU, NW, all: all.concat([P0], L.flatMap(v => [v.U, v.UC].filter(Boolean))) };
    M.grid = grid;
    M.render_ = null;
    if (raster) M.setRaster(raster);
    M.applyParams();
  };

  M.setRaster = function (raster) {
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
    M.device.queue.writeBuffer(M.S.B.dgX, 0, dgX); M.device.queue.writeBuffer(M.S.B.dgZ, 0, dgZ); M.device.queue.writeBuffer(M.S.B.ks, 0, ks);
    M.applyParams();
  };

  // bed height at a column: the solid cells counted up from the floor
  function bedAt(i) {
    const g = M.grid, s = M.solidHost; if (!s) return 0;
    let k = 0; while (k < g.nz && s[i + g.nx * k]) k++; return k * g.dx;
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
    if (M.ok && M.S) writeP(M.S.P0, P);
  };

  // Fill still water to h0 (fractional top cell), zero velocity, time 0.
  M.reset = function () {
    M.t = 0;
    if (!M.ok || !M.S || !M.grid) return;
    const { nx, nz, dx } = M.grid, N = nx * nz, f = new Float32Array(N), sol = M.solidHost || new Uint32Array(N);
    const h0 = M.p.h0;
    for (let k = 0; k < nz; k++) { const fr = Math.min(Math.max((h0 - k * dx) / dx, 0), 1); if (fr <= 0) break; for (let i = 0; i < nx; i++) f[i + nx * k] = sol[i + nx * k] ? 0 : fr; }
    const B = M.S.B, q = M.device.queue;
    q.writeBuffer(B.f, 0, f); q.writeBuffer(B.u, 0, new Float32Array(M.S.NU)); q.writeBuffer(B.w, 0, new Float32Array(M.S.NW)); q.writeBuffer(B.tb, 0, new Float32Array(4));
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
    pass = enc.beginComputePass(); run(pass, d.dragU); run(pass, d.dragW); run(pass, d.bcU); run(pass, d.bcW); run(pass, d.classify); pass.end();
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
      M.t += n * M.dt; M.stepsFrame = n;
      M.busy = true; const t0 = performance.now();
      M.device.queue.onSubmittedWorkDone().then(() => { M.msStep = (performance.now() - t0) / n; M.busy = false; }, () => { M.busy = false; });
    }
    if (M.frameNo % 6 === 0 && !M.readBusy) readScalars();
  };
  function readScalars() {
    M.readBusy = true;
    const st = M.device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = M.device.createCommandEncoder(); e.copyBufferToBuffer(M.S.B.scal, 0, st, 0, 32); M.device.queue.submit([e.finish()]);
    st.mapAsync(GPUMapMode.READ).then(() => {
      const a = new Float32Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy();
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
      const bg = M.device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [U, B.f, B.solid, B.u, B.w, B.eta].map((b, j) => ({ binding: j, resource: { buffer: b } })) });
      M.render_ = { p, U, bg };
    }
    const g = M.grid, r = view.dpr || 1;
    const v = new Float32Array([view.x0, view.z0, view.s * r, view.ve, view.cw * r, view.ch * r, g.dx, g.nx * g.dx, g.nx, g.nz, M.colorMode,
      g.nz * g.dx, M.t, Math.max(M.vmax, 0.05), Math.max(M.p.h0, g.dx), 0]);
    M.device.queue.writeBuffer(M.render_.U, 0, v);
    const enc = M.device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: (view.target || M.ctx.getCurrentTexture()).createView(), loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }] });
    pass.setPipeline(M.render_.p); pass.setBindGroup(0, M.render_.bg); pass.draw(3); pass.end();
    M.device.queue.submit([enc.finish()]);
  };
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
    waveNumber, pistonStroke, SRC, RENDER,
  };

  registerModel(M);
})();
