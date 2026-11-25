// node tests/mlp2d_test.mjs: gradient check, then train on a spiral and measure how close the flow's samples land
import { Trainer, Field, initParams, gauss, paramCount } from '../docs/mlp2d.js';
const spiral = [];
for (let i = 0; i < 2000; i++) { const r = Math.sqrt(Math.random()) * 2.6, a = 3 * r; spiral.push([r * Math.cos(a) + 0.08 * gauss(), r * Math.sin(a) + 0.08 * gauss()]); }
const sData = o => { const q = spiral[(Math.random() * spiral.length) | 0]; o[0] = q[0]; o[1] = q[1]; };
const sInit = o => { o[0] = gauss(); o[1] = gauss(); };

// finite-difference gradient check on a fixed batch
{
  const p = initParams(); const tr = new Trainer(p, 8);
  const pts = Array.from({ length: 64 }, () => [gauss(), gauss()]); let k = 0;
  const det = o => { const q = pts[k++ % 64]; o[0] = q[0]; o[1] = q[1]; };
  const rnd = Math.random; let seed = 1; const lcg = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const lossAt = () => { k = 0; seed = 1; Math.random = lcg; const save = p.slice(); const m = tr.m.slice(), v = tr.v.slice(), kk = tr.k;
    const l = tr.step(det, det, 0); p.set(save); tr.m.set(m); tr.v.set(v); tr.k = kk; Math.random = rnd; return l; };
  lossAt(); const g = tr.g.slice(); let worst = 0;
  for (const i of [0, 5, 700, 3000, paramCount() - 1, paramCount() - 3]) {
    const e = 1e-2, o = p[i]; p[i] = o + e; const lp = lossAt(); p[i] = o - e; const lm = lossAt(); p[i] = o;
    const fd = (lp - lm) / (2 * e); worst = Math.max(worst, Math.abs(fd - g[i]) / (Math.abs(fd) + Math.abs(g[i]) + 1e-6));
  }
  console.log('grad check worst rel err', worst.toExponential(2));
}

const p = initParams(), tr = new Trainer(p, +process.argv[2] || 256), f = new Field(); f.set(p);
const N = +process.argv[3] || 3000; let t0 = performance.now(), ema = null;
for (let k = 0; k < N; k++) {
  const lr = 2e-3 * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * k / N)));
  const l = tr.step(sData, sInit, lr); ema = ema === null ? l : 0.98 * ema + 0.02 * l;
  if (k % 500 === 0 || k === N - 1) console.log(k, ema.toFixed(4), ((performance.now() - t0) / (k + 1)).toFixed(2), 'ms/step');
}
// push 1000 noise points through 60 Euler steps; mean distance to nearest data point, vs raw noise
const d = (x, y) => Math.min(...spiral.map(q => Math.hypot(q[0] - x, q[1] - y)));
let m0 = 0, m1 = 0; const u = new Float32Array(2);
for (let i = 0; i < 1000; i++) {
  let x = gauss(), y = gauss(); m0 += d(x, y);
  for (let s = 0; s < 60; s++) { f.eval(x, y, s / 60, u); x += u[0] / 60; y += u[1] / 60; }
  m1 += d(x, y);
}
console.log('mean dist to spiral: noise', (m0 / 1000).toFixed(3), ' flow samples', (m1 / 1000).toFixed(3));
