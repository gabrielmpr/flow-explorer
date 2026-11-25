// Trains the 2-D flow off the main thread. Every few steps it posts the current weights so the page
// can draw the field and move particles with them.
import { Trainer, initParams, gauss } from './mlp2d.js';

const JITTER = 0.04;  // each painted point stands for a small Gaussian blob
let job = 0;

self.onmessage = e => {
  const { data, init, steps, snaps } = e.data;
  const id = ++job;
  const params = initParams();
  const tr = new Trainer(params, 128);
  const pick = (pts, o) => {
    const i = (Math.random() * (pts.length / 2)) | 0;
    o[0] = pts[2 * i] + JITTER * gauss(); o[1] = pts[2 * i + 1] + JITTER * gauss();
  };
  const sData = o => pick(data, o);
  const sInit = init ? (o => pick(init, o)) : (o => { o[0] = gauss(); o[1] = gauss(); });
  let k = 0, ema = null;
  const snapSet = new Set(snaps);
  self.postMessage({ id, step: 0, loss: null, params: params.slice(), snap: snapSet.has(0) });
  const run = () => {
    if (id !== job) return;              // a newer job replaced this one
    const t0 = performance.now();
    while (k < steps && performance.now() - t0 < 30) {
      const lr = 2e-2 * (0.05 + 0.95 * 0.5 * (1 + Math.cos(Math.PI * k / steps)));
      const l = tr.step(sData, sInit, lr);
      ema = ema === null ? l : 0.97 * ema + 0.03 * l;
      k++;
      if (snapSet.has(k)) self.postMessage({ id, step: k, loss: ema, params: params.slice(), snap: true });
    }
    self.postMessage({ id, step: k, loss: ema, params: params.slice(), snap: false, done: k >= steps });
    if (k < steps) setTimeout(run, 0);
  };
  setTimeout(run, 0);
};
