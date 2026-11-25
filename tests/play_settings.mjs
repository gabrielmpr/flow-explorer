import { Trainer, Field, initParams, gauss } from '../docs/mlp2d.js';
const [B, LR, STEPS, width] = [+process.argv[2] || 128, +process.argv[3] || 3e-3, +process.argv[4] || 2000, +process.argv[5] || 0.04];
const data = []; for (let i = 0; i < 1500; i++) { const r = Math.sqrt(Math.random()) * 2.7, a = 2.8 * r + Math.PI * (Math.random() < 0.5); data.push([r * Math.cos(a), r * Math.sin(a)]); }
const sD = o => { const q = data[(Math.random() * data.length) | 0]; o[0] = q[0] + width * gauss(); o[1] = q[1] + width * gauss(); };
const sI = o => { o[0] = gauss(); o[1] = gauss(); };
const p = initParams(), tr = new Trainer(p, B), f = new Field(); f.set(p); const u = new Float32Array(2);
const starts = Array.from({ length: 500 }, () => [gauss(), gauss()]);
const dist = () => { let m = 0; for (const [x0, y0] of starts) { let x = x0, y = y0; for (let s = 0; s < 30; s++) { f.eval(x, y, s / 30, u); x += u[0] / 30; y += u[1] / 30; } m += Math.min(...data.map(q => Math.hypot(q[0] - x, q[1] - y))); } return (m / starts.length).toFixed(3); };
const t0 = performance.now(); const out = [];
for (let k = 0; k <= STEPS; k++) {
  if ([0, 200, 500, 1000, 2000, 4000].includes(k)) out.push(`${k}:${dist()}`);
  if (k === STEPS) break;
  tr.step(sD, sI, LR * (0.05 + 0.95 * 0.5 * (1 + Math.cos(Math.PI * k / STEPS))));
}
console.log(`B=${B} lr=${LR} steps=${STEPS} w=${width}`, out.join(' '), ((performance.now() - t0) / STEPS).toFixed(1) + 'ms');
