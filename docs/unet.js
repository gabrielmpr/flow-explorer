// The U-Net from model.py, run in the browser. Two backends with the same interface:
//   GpuUNet: WebGPU compute shaders, one dispatch per layer.
//   CpuUNet: plain JavaScript loops, used when WebGPU is missing and as the reference for the GPU path.
// forward(x, t, B) takes x as a Float32Array of B*3*32*32 values (NCHW) and t as B times, returns u_theta(x, t).

export const S = 32; // image side

export async function loadWeights(base = '.') {
  const [cfg, buf] = await Promise.all([
    fetch(`${base}/model.json`).then(r => r.json()),
    fetch(`${base}/model.bin`).then(r => r.arrayBuffer()),
  ]);
  const h = new Uint16Array(buf);
  const W = new Map();
  for (const t of cfg.tensors) {
    const a = new Float32Array(t.size);
    for (let i = 0; i < t.size; i++) a[i] = f16(h[t.offset + i]);
    W.set(t.name, { data: a, shape: t.shape });
  }
  return { W, nCh: cfg.n_ch, step: cfg.step };
}

function f16(b) {
  const s = b & 0x8000 ? -1 : 1, e = (b >> 10) & 0x1f, m = b & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

const BLOCKS = ['down1', 'down2', 'down3', 'mid', 'up3', 'up2', 'up1'];

// time embedding and each block's per-channel time bias; tiny, so both backends do it on the CPU
function timeBiases(W, nCh, t, B) {
  const half = nCh / 2, td = 4 * nCh;
  const emb = new Float32Array(B * nCh);
  for (let b = 0; b < B; b++)
    for (let i = 0; i < half; i++) {
      const f = Math.exp(Math.log(1000) * i / (half - 1)) * t[b];
      emb[b * nCh + i] = Math.sin(f);
      emb[b * nCh + half + i] = Math.cos(f);
    }
  const h = linear(emb, W.get('time_emb.mlp.0.weight'), W.get('time_emb.mlp.0.bias'), B);
  for (let i = 0; i < h.length; i++) h[i] = h[i] / (1 + Math.exp(-h[i]));
  const te = linear(h, W.get('time_emb.mlp.2.weight'), W.get('time_emb.mlp.2.bias'), B);
  const out = {};
  for (const k of BLOCKS) out[k] = linear(te, W.get(`${k}.t_proj.weight`), W.get(`${k}.t_proj.bias`), B);
  return out;
}

function linear(x, w, bias, B) {
  const [o, i] = w.shape, y = new Float32Array(B * o);
  for (let b = 0; b < B; b++)
    for (let r = 0; r < o; r++) {
      let s = bias.data[r];
      for (let c = 0; c < i; c++) s += w.data[r * i + c] * x[b * i + c];
      y[b * o + r] = s;
    }
  return y;
}

// ---------------------------------------------------------------- CPU

export class CpuUNet {
  constructor(W, nCh) { this.W = W; this.nCh = nCh; this.kind = 'cpu'; }

  conv(x, B, C, H, Wd, name, tb = null, res = null) {
    const w = this.W.get(`${name}.weight`), bias = this.W.get(`${name}.bias`).data;
    const [Co, Ci, K] = w.shape, r = K >> 1, HW = H * Wd, wd = w.data;
    const y = new Float32Array(B * Co * HW);
    for (let b = 0; b < B; b++)
      for (let co = 0; co < Co; co++) {
        const ob = (b * Co + co) * HW;
        const bv = bias[co] + (tb ? tb[b * Co + co] : 0);
        for (let i = 0; i < HW; i++) y[ob + i] = bv;
        for (let ci = 0; ci < Ci; ci++) {
          const ib = (b * Ci + ci) * HW;
          for (let ky = 0; ky < K; ky++)
            for (let kx = 0; kx < K; kx++) {
              const wv = wd[((co * Ci + ci) * K + ky) * K + kx], dy = ky - r, dx = kx - r;
              const y0 = Math.max(0, -dy), y1 = Math.min(H, H - dy), x0 = Math.max(0, -dx), x1 = Math.min(Wd, Wd - dx);
              for (let yy = y0; yy < y1; yy++) {
                const orow = ob + yy * Wd, irow = ib + (yy + dy) * Wd + dx;
                for (let xx = x0; xx < x1; xx++) y[orow + xx] += wv * x[irow + xx];
              }
            }
        }
        if (res) for (let i = 0; i < HW; i++) y[ob + i] += res[ob + i];
      }
    return y;
  }

  gnSilu(x, B, C, HW, name) {
    const g = this.W.get(`${name}.weight`).data, be = this.W.get(`${name}.bias`).data;
    const Cg = C / 8, y = new Float32Array(x.length);
    for (let b = 0; b < B; b++)
      for (let gr = 0; gr < 8; gr++) {
        const s0 = (b * C + gr * Cg) * HW, n = Cg * HW;
        let m = 0; for (let i = 0; i < n; i++) m += x[s0 + i]; m /= n;
        let v = 0; for (let i = 0; i < n; i++) { const d = x[s0 + i] - m; v += d * d; } v /= n;
        const rs = 1 / Math.sqrt(v + 1e-5);
        for (let c = 0; c < Cg; c++) {
          const ch = gr * Cg + c, a = g[ch] * rs, o = be[ch] - m * a, base = s0 + c * HW;
          for (let i = 0; i < HW; i++) { const z = x[base + i] * a + o; y[base + i] = z / (1 + Math.exp(-z)); }
        }
      }
    return y;
  }

  pool(x, B, C, H) {
    const h = H / 2, y = new Float32Array(B * C * h * h);
    for (let bc = 0; bc < B * C; bc++)
      for (let i = 0; i < h; i++)
        for (let j = 0; j < h; j++) {
          const s = bc * H * H + 2 * i * H + 2 * j;
          y[bc * h * h + i * h + j] = 0.25 * (x[s] + x[s + 1] + x[s + H] + x[s + H + 1]);
        }
    return y;
  }

  upcat(a, Ca, skip, Cb, B, H) { // a is (B,Ca,H/2,H/2), skip is (B,Cb,H,H)
    const h = H / 2, C = Ca + Cb, HW = H * H, y = new Float32Array(B * C * HW);
    for (let b = 0; b < B; b++) {
      for (let c = 0; c < Ca; c++)
        for (let i = 0; i < H; i++)
          for (let j = 0; j < H; j++) y[(b * C + c) * HW + i * H + j] = a[(b * Ca + c) * h * h + (i >> 1) * h + (j >> 1)];
      y.set(skip.subarray(b * Cb * HW, (b + 1) * Cb * HW), (b * C + Ca) * HW);
    }
    return y;
  }

  block(x, B, Ci, Co, H, name, tb) {
    const n1 = this.gnSilu(x, B, Ci, H * H, `${name}.norm1`);
    const h = this.conv(n1, B, Ci, H, H, `${name}.conv1`, tb);
    const n2 = this.gnSilu(h, B, Co, H * H, `${name}.norm2`);
    const s = this.W.has(`${name}.skip.weight`) ? this.conv(x, B, Ci, H, H, `${name}.skip`) : x;
    return this.conv(n2, B, Co, H, H, `${name}.conv2`, null, s);
  }

  async forward(x, t, B) {
    const c = this.nCh, tb = timeBiases(this.W, c, t, B);
    const h1 = this.block(this.conv(x, B, 3, S, S, 'inp'), B, c, c, S, 'down1', tb.down1);
    const h2 = this.block(this.pool(h1, B, c, S), B, c, 2 * c, S / 2, 'down2', tb.down2);
    const h3 = this.block(this.pool(h2, B, 2 * c, S / 2), B, 2 * c, 4 * c, S / 4, 'down3', tb.down3);
    let h = this.block(this.pool(h3, B, 4 * c, S / 4), B, 4 * c, 4 * c, S / 8, 'mid', tb.mid);
    h = this.block(this.upcat(h, 4 * c, h3, 4 * c, B, S / 4), B, 8 * c, 2 * c, S / 4, 'up3', tb.up3);
    h = this.block(this.upcat(h, 2 * c, h2, 2 * c, B, S / 2), B, 4 * c, c, S / 2, 'up2', tb.up2);
    h = this.block(this.upcat(h, c, h1, c, B, S), B, 2 * c, c, S, 'up1', tb.up1);
    return this.conv(this.gnSilu(h, B, c, S * S, 'out.0'), B, c, S, S, 'out.2');
  }
}

// ---------------------------------------------------------------- WebGPU

const WGSL_CONV = /* wgsl */`
struct P { B: u32, Ci: u32, Co: u32, H: u32, W: u32, K: u32, flags: u32, Co4: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<vec4f>;    // (Ci, K, K, Co4) groups of 4 output channels
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> tb: array<f32>;     // (B, Co), used if flags & 1
@group(0) @binding(5) var<storage, read> res: array<f32>;    // (B, Co, H, W), used if flags & 2
@group(0) @binding(6) var<storage, read_write> outp: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let HW = p.H * p.W;
  if (g.x >= HW) { return; }
  let b = g.z / p.Co4;
  let q = g.z % p.Co4;
  let y = i32(g.x / p.W);
  let x = i32(g.x % p.W);
  let r = i32(p.K / 2u);
  var acc = vec4f(0.0);
  for (var ci = 0u; ci < p.Ci; ci++) {
    let ib = (b * p.Ci + ci) * HW;
    for (var ky = 0; ky < i32(p.K); ky++) {
      let yy = y + ky - r;
      if (yy < 0 || yy >= i32(p.H)) { continue; }
      for (var kx = 0; kx < i32(p.K); kx++) {
        let xx = x + kx - r;
        if (xx < 0 || xx >= i32(p.W)) { continue; }
        let v = inp[ib + u32(yy) * p.W + u32(xx)];
        acc += v * w[((ci * p.K + u32(ky)) * p.K + u32(kx)) * p.Co4 + q];
      }
    }
  }
  for (var j = 0u; j < 4u; j++) {
    let co = q * 4u + j;
    if (co >= p.Co) { break; }
    var v = acc[j] + bias[co];
    if ((p.flags & 1u) != 0u) { v += tb[b * p.Co + co]; }
    let o = (b * p.Co + co) * HW + g.x;
    if ((p.flags & 2u) != 0u) { v += res[o]; }
    outp[o] = v;
  }
}`;

// 3x3 conv, each thread computes 4 adjacent pixels of one row for 4 output channels:
// the 6 input values of a row segment are loaded once and reused across the 3 taps and 4 pixels.
const WGSL_CONV3 = /* wgsl */`
struct P { B: u32, Ci: u32, Co: u32, H: u32, W: u32, K: u32, flags: u32, Co4: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<vec4f>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> tb: array<f32>;
@group(0) @binding(5) var<storage, read> res: array<f32>;
@group(0) @binding(6) var<storage, read_write> outp: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let Wq = p.W / 4u;
  if (g.x >= p.H * Wq) { return; }
  let HW = p.H * p.W;
  let b = g.z / p.Co4;
  let q = g.z % p.Co4;
  let y = i32(g.x / Wq);
  let x0 = (g.x % Wq) * 4u;
  let hasL = x0 > 0u;
  let hasR = x0 + 4u < p.W;
  var a0 = vec4f(0.0); var a1 = vec4f(0.0); var a2 = vec4f(0.0); var a3 = vec4f(0.0);
  for (var ci = 0u; ci < p.Ci; ci++) {
    let ib = (b * p.Ci + ci) * HW;
    for (var ky = 0u; ky < 3u; ky++) {
      let yy = y + i32(ky) - 1;
      if (yy < 0 || yy >= i32(p.H)) { continue; }
      let row = ib + u32(yy) * p.W + x0;
      let v0 = select(0.0, inp[select(row, row - 1u, hasL)], hasL);
      let v1 = inp[row]; let v2 = inp[row + 1u]; let v3 = inp[row + 2u]; let v4 = inp[row + 3u];
      let v5 = select(0.0, inp[select(row, row + 4u, hasR)], hasR);
      let wb = (ci * 3u + ky) * 3u * p.Co4 + q;
      let w0 = w[wb]; let w1 = w[wb + p.Co4]; let w2 = w[wb + 2u * p.Co4];
      a0 += v0 * w0 + v1 * w1 + v2 * w2;
      a1 += v1 * w0 + v2 * w1 + v3 * w2;
      a2 += v2 * w0 + v3 * w1 + v4 * w2;
      a3 += v3 * w0 + v4 * w1 + v5 * w2;
    }
  }
  let acc = array<vec4f, 4>(a0, a1, a2, a3);
  for (var j = 0u; j < 4u; j++) {
    let co = q * 4u + j;
    if (co >= p.Co) { break; }
    var base = bias[co];
    if ((p.flags & 1u) != 0u) { base += tb[b * p.Co + co]; }
    let o = (b * p.Co + co) * HW + u32(y) * p.W + x0;
    for (var i = 0u; i < 4u; i++) {
      var v = acc[i][j] + base;
      if ((p.flags & 2u) != 0u) { v += res[o + i]; }
      outp[o + i] = v;
    }
  }
}`;

const WGSL_CONV3x8 = /* wgsl */`
struct P { B: u32, Ci: u32, Co: u32, H: u32, W: u32, K: u32, flags: u32, Co4: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<vec4f>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> tb: array<f32>;
@group(0) @binding(5) var<storage, read> res: array<f32>;
@group(0) @binding(6) var<storage, read_write> outp: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let Wq = p.W / 4u;
  if (g.x >= p.H * Wq) { return; }
  let HW = p.H * p.W;
  let b = g.z / (p.Co4 / 2u);
  let q = (g.z % (p.Co4 / 2u)) * 2u;
  let y = i32(g.x / Wq);
  let x0 = (g.x % Wq) * 4u;
  let hasL = x0 > 0u;
  let hasR = x0 + 4u < p.W;
  var a0 = vec4f(0.0); var a1 = vec4f(0.0); var a2 = vec4f(0.0); var a3 = vec4f(0.0);
  var c0 = vec4f(0.0); var c1 = vec4f(0.0); var c2 = vec4f(0.0); var c3 = vec4f(0.0);
  for (var ci = 0u; ci < p.Ci; ci++) {
    let ib = (b * p.Ci + ci) * HW;
    for (var ky = 0u; ky < 3u; ky++) {
      let yy = y + i32(ky) - 1;
      if (yy < 0 || yy >= i32(p.H)) { continue; }
      let row = ib + u32(yy) * p.W + x0;
      let v0 = select(0.0, inp[select(row, row - 1u, hasL)], hasL);
      let v1 = inp[row]; let v2 = inp[row + 1u]; let v3 = inp[row + 2u]; let v4 = inp[row + 3u];
      let v5 = select(0.0, inp[select(row, row + 4u, hasR)], hasR);
      let wb = (ci * 3u + ky) * 3u * p.Co4 + q;
      let w0 = w[wb]; let w1 = w[wb + p.Co4]; let w2 = w[wb + 2u * p.Co4];
      a0 += v0 * w0 + v1 * w1 + v2 * w2;
      a1 += v1 * w0 + v2 * w1 + v3 * w2;
      a2 += v2 * w0 + v3 * w1 + v4 * w2;
      a3 += v3 * w0 + v4 * w1 + v5 * w2;
      let e0 = w[wb + 1u]; let e1 = w[wb + p.Co4 + 1u]; let e2 = w[wb + 2u * p.Co4 + 1u];
      c0 += v0 * e0 + v1 * e1 + v2 * e2;
      c1 += v1 * e0 + v2 * e1 + v3 * e2;
      c2 += v2 * e0 + v3 * e1 + v4 * e2;
      c3 += v3 * e0 + v4 * e1 + v5 * e2;
    }
  }
  let acc = array<vec4f, 4>(a0, a1, a2, a3);
  let acc2 = array<vec4f, 4>(c0, c1, c2, c3);
  for (var j = 0u; j < 8u; j++) {
    let co = q * 4u + j;
    var base = bias[co];
    if ((p.flags & 1u) != 0u) { base += tb[b * p.Co + co]; }
    let o = (b * p.Co + co) * HW + u32(y) * p.W + x0;
    for (var i = 0u; i < 4u; i++) {
      var v = select(acc2[i][j % 4u], acc[i][j % 4u], j < 4u) + base;
      if ((p.flags & 2u) != 0u) { v += res[o + i]; }
      outp[o + i] = v;
    }
  }
}`;

const WGSL_GN = /* wgsl */`
struct P { C: u32, HW: u32, a: u32, b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
var<workgroup> red: array<f32, 256>;

fn reduce(l: u32, v: f32) -> f32 {
  red[l] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (l < s) { red[l] += red[l + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = wg.x / 8u;
  let gr = wg.x % 8u;
  let Cg = p.C / 8u;
  let n = Cg * p.HW;
  let s0 = (b * p.C + gr * Cg) * p.HW;
  var s = 0.0;
  for (var i = l; i < n; i += 256u) { s += inp[s0 + i]; }
  let m = reduce(l, s) / f32(n);
  var v = 0.0;
  for (var i = l; i < n; i += 256u) { let d = inp[s0 + i] - m; v += d * d; }
  let rs = inverseSqrt(reduce(l, v) / f32(n) + 1e-5);
  for (var i = l; i < n; i += 256u) {
    let ch = gr * Cg + i / p.HW;
    let z = (inp[s0 + i] - m) * rs * gamma[ch] + beta[ch];
    outp[s0 + i] = z / (1.0 + exp(-z));
  }
}`;

const WGSL_POOL = /* wgsl */`
struct P { n: u32, H: u32, a: u32, b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + g.y * 65535u * 64u;
  if (i >= p.n) { return; }
  let h = p.H / 2u;
  let bc = i / (h * h);
  let r = i % (h * h);
  let s = bc * p.H * p.H + 2u * (r / h) * p.H + 2u * (r % h);
  outp[i] = 0.25 * (inp[s] + inp[s + 1u] + inp[s + p.H] + inp[s + p.H + 1u]);
}`;

const WGSL_UPCAT = /* wgsl */`
struct P { n: u32, H: u32, Ca: u32, Cb: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> sk: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + g.y * 65535u * 64u;
  if (i >= p.n) { return; }
  let HW = p.H * p.H;
  let C = p.Ca + p.Cb;
  let b = i / (C * HW);
  let c = (i / HW) % C;
  let r = i % HW;
  if (c < p.Ca) {
    let h = p.H / 2u;
    outp[i] = a[(b * p.Ca + c) * h * h + ((r / p.H) / 2u) * h + (r % p.H) / 2u];
  } else {
    outp[i] = sk[(b * p.Cb + c - p.Ca) * HW + r];
  }
}`;

export class GpuUNet {
  static async create(W, nCh) {
    if (!navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    return new GpuUNet(device, W, nCh);
  }

  constructor(device, W, nCh) {
    this.kind = 'gpu';
    this.device = device; this.W = W; this.nCh = nCh;
    const mk = code => device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
    this.pipes = { conv: mk(WGSL_CONV), conv3: mk(WGSL_CONV3), conv3x8: mk(WGSL_CONV3x8), gn: mk(WGSL_GN), pool: mk(WGSL_POOL), upcat: mk(WGSL_UPCAT) };
    this.wbuf = new Map();
    for (const [name, t] of W) {
      if (name.startsWith('time_emb') || name.includes('t_proj')) continue;
      let data = t.data;
      if (t.shape.length === 4) { // (Co, Ci, K, K) -> (Ci, K, K, Co padded to 4)
        const [Co, Ci, K] = t.shape, Co4 = Math.ceil(Co / 4) * 4;
        data = new Float32Array(Ci * K * K * Co4);
        for (let co = 0; co < Co; co++)
          for (let ci = 0; ci < Ci; ci++)
            for (let k = 0; k < K * K; k++) data[(ci * K * K + k) * Co4 + co] = t.data[(co * Ci + ci) * K * K + k];
      }
      this.wbuf.set(name, this.storage(data));
    }
    this.dummy = this.storage(new Float32Array(4));
    this.plans = new Map();
  }

  storage(data, extra = 0) {
    const buf = this.device.createBuffer({ size: Math.max(16, data ? data.byteLength : extra), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    if (data) this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  uniform(arr) {
    const buf = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, new Uint32Array([...arr, 0, 0, 0, 0, 0, 0, 0, 0].slice(0, 8)));
    return buf;
  }

  // Build every buffer, bind group and dispatch size for batch size B once; reuse it on every call.
  plan(B) {
    if (this.plans.has(B)) return this.plans.get(B);
    const c = this.nCh, ops = [], tbBufs = {};
    const act = n => this.storage(null, n * 4);
    const bg = (pipe, bufs) => this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const lin = n => { const gx = Math.ceil(n / 64); return [Math.min(gx, 65535), Math.ceil(gx / 65535), 1]; };

    const conv = (x, Ci, Co, H, name, tb = null, res = null) => {
      const K = this.W.get(`${name}.weight`).shape[2], Co4 = Math.ceil(Co / 4);
      const y = act(B * Co * H * H);
      const u = this.uniform([B, Ci, Co, H, H, K, (tb ? 1 : 0) | (res ? 2 : 0), Co4]);
      const wide = K === 3 && Co % 8 === 0, pipe = K === 3 ? (wide ? this.pipes.conv3x8 : this.pipes.conv3) : this.pipes.conv, px = K === 3 ? H * H / 4 : H * H;
      ops.push([pipe, bg(pipe, [u, x, this.wbuf.get(`${name}.weight`), this.wbuf.get(`${name}.bias`), tb || this.dummy, res || this.dummy, y]),
        [Math.ceil(px / 64), 1, B * (wide ? Co4 / 2 : Co4)]]);
      return y;
    };
    const gn = (x, C, H, name) => {
      const y = act(B * C * H * H);
      ops.push([this.pipes.gn, bg(this.pipes.gn, [this.uniform([C, H * H]), x, this.wbuf.get(`${name}.weight`), this.wbuf.get(`${name}.bias`), y]), [B * 8, 1, 1]]);
      return y;
    };
    const pool = (x, C, H) => {
      const n = B * C * H * H / 4, y = act(n);
      ops.push([this.pipes.pool, bg(this.pipes.pool, [this.uniform([n, H]), x, y]), lin(n)]);
      return y;
    };
    const upcat = (a, Ca, sk, Cb, H) => {
      const n = B * (Ca + Cb) * H * H, y = act(n);
      ops.push([this.pipes.upcat, bg(this.pipes.upcat, [this.uniform([n, H, Ca, Cb]), a, sk, y]), lin(n)]);
      return y;
    };
    const block = (x, Ci, Co, H, name) => {
      tbBufs[name] = act(B * Co);
      const h = conv(gn(x, Ci, H, `${name}.norm1`), Ci, Co, H, `${name}.conv1`, tbBufs[name]);
      const s = this.W.has(`${name}.skip.weight`) ? conv(x, Ci, Co, H, `${name}.skip`) : x;
      return conv(gn(h, Co, H, `${name}.norm2`), Co, Co, H, `${name}.conv2`, null, s);
    };

    const xin = act(B * 3 * S * S);
    const h1 = block(conv(xin, 3, c, S, 'inp'), c, c, S, 'down1');
    const h2 = block(pool(h1, c, S), c, 2 * c, S / 2, 'down2');
    const h3 = block(pool(h2, 2 * c, S / 2), 2 * c, 4 * c, S / 4, 'down3');
    let h = block(pool(h3, 4 * c, S / 4), 4 * c, 4 * c, S / 8, 'mid');
    h = block(upcat(h, 4 * c, h3, 4 * c, S / 4), 8 * c, 2 * c, S / 4, 'up3');
    h = block(upcat(h, 2 * c, h2, 2 * c, S / 2), 4 * c, c, S / 2, 'up2');
    h = block(upcat(h, c, h1, c, S), 2 * c, c, S, 'up1');
    const yout = conv(gn(h, c, S, 'out.0'), c, 3, S, 'out.2');
    const read = this.device.createBuffer({ size: B * 3 * S * S * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const plan = { ops, xin, yout, read, tbBufs };
    this.plans.set(B, plan);
    return plan;
  }

  async forward(x, t, B) {
    const P = this.plan(B), q = this.device.queue;
    q.writeBuffer(P.xin, 0, x);
    const tb = timeBiases(this.W, this.nCh, t, B);
    for (const k of BLOCKS) q.writeBuffer(P.tbBufs[k], 0, tb[k]);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (const [pipe, group, [gx, gy, gz]] of P.ops) { pass.setPipeline(pipe); pass.setBindGroup(0, group); pass.dispatchWorkgroups(gx, gy, gz); }
    pass.end();
    enc.copyBufferToBuffer(P.yout, 0, P.read, 0, B * 3 * S * S * 4);
    q.submit([enc.finish()]);
    await P.read.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(P.read.getMappedRange().slice(0));
    P.read.unmap();
    return out;
  }
}

export async function createEngine(base = '.') {
  const { W, nCh, step } = await loadWeights(base);
  let net = null;
  try { net = await GpuUNet.create(W, nCh); } catch (e) { console.warn('WebGPU unavailable', e); }
  return { net: net || new CpuUNet(W, nCh), W, nCh, step };
}
