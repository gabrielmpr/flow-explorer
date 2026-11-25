// A small MLP u_theta(x, y, t) -> (u_x, u_y) with hand-written backprop and Adam.
// Trained with the straight-line flow matching loss: x_t = (1 - t) x0 + t x1, target x1 - x0.

export const XF = [0.5, 1, 2, 4];  // frequencies for sin/cos features of x and y
export const NF = 10 + 4 * XF.length;  // x, y, t, 1 - t, sin/cos(pi k t) for k = 1..3, sin/cos(f x), sin/cos(f y)
export const HID = 64;
export const DEPTH = 3;        // hidden layers

export function features(x, y, t, out, o = 0) {
  out[o] = x; out[o + 1] = y; out[o + 2] = t; out[o + 3] = 1 - t;
  for (let k = 1; k <= 3; k++) {
    out[o + 2 + 2 * k] = Math.sin(Math.PI * k * t);
    out[o + 3 + 2 * k] = Math.cos(Math.PI * k * t);
  }
  let j = o + 10;
  for (const f of XF) {
    out[j++] = Math.sin(f * x); out[j++] = Math.cos(f * x);
    out[j++] = Math.sin(f * y); out[j++] = Math.cos(f * y);
  }
}

function layerSizes() {
  const s = [NF];
  for (let i = 0; i < DEPTH; i++) s.push(HID);
  s.push(2);
  return s;
}

// all parameters live in one Float32Array so the worker can post them in one message
export function paramCount() {
  const s = layerSizes();
  let n = 0;
  for (let l = 0; l + 1 < s.length; l++) n += s[l] * s[l + 1] + s[l + 1];
  return n;
}

export function initParams(rand = Math.random) {
  const s = layerSizes(), p = new Float32Array(paramCount());
  let o = 0;
  for (let l = 0; l + 1 < s.length; l++) {
    const a = Math.sqrt(1 / s[l]) * (l + 2 === s.length ? 0.1 : 1);
    for (let i = 0; i < s[l] * s[l + 1]; i++) p[o++] = (2 * rand() - 1) * a * Math.sqrt(3);
    o += s[l + 1]; // biases start at 0
  }
  return p;
}

// Evaluate the field at one point. Allocation-free; used per particle per frame.
export class Field {
  constructor() {
    this.sizes = layerSizes();
    this.bufs = this.sizes.map(w => new Float32Array(w));
    this.params = null;
  }
  set(p) { this.params = p; }
  eval(x, y, t, out) {
    const p = this.params, s = this.sizes, L = s.length - 1;
    features(x, y, t, this.bufs[0]);
    let o = 0;
    for (let l = 0; l < L; l++) {
      const ni = s[l], no = s[l + 1], bo = o + ni * no, A = this.bufs[l], Y = l + 1 === L ? out : this.bufs[l + 1];
      for (let j = 0; j < no; j++) {
        let z = p[bo + j];
        const r = o + j * ni;
        for (let i = 0; i < ni; i++) z += p[r + i] * A[i];
        Y[j] = l + 1 === L ? z : z / (1 + Math.exp(-z));
      }
      o = bo + no;
    }
    return out;
  }
}

export class Trainer {
  constructor(params, batch = 256) {
    this.p = params;
    this.B = batch;
    this.sizes = layerSizes();
    const n = params.length;
    this.g = new Float32Array(n); this.m = new Float32Array(n); this.v = new Float32Array(n);
    this.k = 0;
    this.acts = this.sizes.map(w => new Float32Array(batch * w));   // post-activation, acts[0] = features
    this.pre = this.sizes.map(w => new Float32Array(batch * w));
    this.delta = this.sizes.map(w => new Float32Array(batch * w));
    this.target = new Float32Array(batch * 2);
  }

  // sampleData(out2) and sampleInit(out2) write one (x, y) pair each
  step(sampleData, sampleInit, lr) {
    const { B, sizes, acts, pre, delta, target, p, g } = this;
    const L = sizes.length - 1, x0 = [0, 0], x1 = [0, 0];
    for (let b = 0; b < B; b++) {
      sampleInit(x0); sampleData(x1);
      const t = Math.random();
      features((1 - t) * x0[0] + t * x1[0], (1 - t) * x0[1] + t * x1[1], t, acts[0], b * NF);
      target[2 * b] = x1[0] - x0[0]; target[2 * b + 1] = x1[1] - x0[1];
    }
    // forward
    let o = 0;
    const offs = [];
    for (let l = 0; l < L; l++) {
      const ni = sizes[l], no = sizes[l + 1], bo = o + ni * no, A = acts[l], Z = pre[l + 1], Y = acts[l + 1];
      offs.push(o);
      for (let b = 0; b < B; b++)
        for (let j = 0; j < no; j++) {
          let z = p[bo + j];
          const r = o + j * ni, ai = b * ni;
          for (let i = 0; i < ni; i++) z += p[r + i] * A[ai + i];
          Z[b * no + j] = z;
          Y[b * no + j] = l + 1 === L ? z : z / (1 + Math.exp(-z));
        }
      o = bo + no;
    }
    // loss = mean over batch and both coordinates of the squared error
    let loss = 0;
    const out = acts[L], dOut = delta[L];
    for (let i = 0; i < B * 2; i++) { const d = out[i] - target[i]; loss += d * d; dOut[i] = 2 * d / (B * 2); }
    loss /= B * 2;
    // backward
    g.fill(0);
    for (let l = L - 1; l >= 0; l--) {
      const ni = sizes[l], no = sizes[l + 1], o2 = offs[l], bo = o2 + ni * no, A = acts[l], D = delta[l + 1];
      if (l + 1 < L) { // through SiLU: d/dz [z sigma(z)] = s + z s (1 - s)
        const Z = pre[l + 1];
        for (let i = 0; i < B * no; i++) { const s = 1 / (1 + Math.exp(-Z[i])); D[i] *= s * (1 + Z[i] * (1 - s)); }
      }
      const Dp = delta[l];
      if (l > 0) Dp.fill(0);
      for (let b = 0; b < B; b++)
        for (let j = 0; j < no; j++) {
          const d = D[b * no + j];
          if (d === 0) continue;
          g[bo + j] += d;
          const r = o2 + j * ni, ai = b * ni;
          for (let i = 0; i < ni; i++) g[r + i] += d * A[ai + i];
          if (l > 0) for (let i = 0; i < ni; i++) Dp[ai + i] += d * p[r + i];
        }
    }
    // Adam
    this.k++;
    const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** this.k, c2 = 1 - b2 ** this.k;
    for (let i = 0; i < p.length; i++) {
      this.m[i] = b1 * this.m[i] + (1 - b1) * g[i];
      this.v[i] = b2 * this.v[i] + (1 - b2) * g[i] * g[i];
      p[i] -= lr * (this.m[i] / c1) / (Math.sqrt(this.v[i] / c2) + 1e-8);
    }
    return loss;
  }
}

export function gauss() {
  let u = 0;
  while (u === 0) u = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}
