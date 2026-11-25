// The live dog: develop noise into a dog, scrub through it, re-roll the rest from any frame,
// and look up the closest training photos.
import { createEngine, S } from './unet.js';
import { putImage } from './draw.js';

const D = 3 * S * S;
const SIGMA_REROLL = 1.0;   // same setting as Finding 1
const N_REROLL = 9;

function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function gaussFill(arr, rand = Math.random) {
  for (let i = 0; i < arr.length; i += 2) {
    const u = 1 - rand(), v = rand(), r = Math.sqrt(-2 * Math.log(u));
    arr[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < arr.length) arr[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return arr;
}
const wait = ms => new Promise(r => setTimeout(r, ms));
const frameWait = () => new Promise(r => requestAnimationFrame(r));

// one Euler(-Maruyama) step for B images; noise level sigma * sqrt(1 - t), as in model.py
function step(x, u, t, h, sigma, B) {
  if (sigma > 0) {
    const z = gaussFill(new Float32Array(B * D)), a = sigma * Math.sqrt((1 - t) * h), c = 0.5 * sigma * sigma;
    for (let i = 0; i < x.length; i++) x[i] += h * (u[i] + c * (t * u[i] - x[i])) + a * z[i];
  } else for (let i = 0; i < x.length; i++) x[i] += h * u[i];
}

export async function initDogs(root, data) {
  const $ = s => root.querySelector(s);
  const engineNote = $('#d-engine');
  engineNote.textContent = 'Loading the model (7.7 MB)…';
  const { net, step: trainedSteps } = await createEngine('.');
  engineNote.textContent = net.kind === 'gpu'
    ? 'Running on your GPU with WebGPU.'
    : 'WebGPU is not available in this browser, so the model runs on the CPU at about one second per step. Fewer steps are set by default.';

  let N = net.kind === 'gpu' ? 50 : 10, sigma = 0, seed = 0, frames = [], sel = 0, job = 0, rerollJob = 0;
  const xtCv = $('#d-xt'), guessCv = $('#d-guess'), slider = $('#d-t'), tval = $('#d-tval');
  const cells = [...root.querySelectorAll('#d-strip .cell')];
  const sheet = [...root.querySelectorAll('#d-sheet canvas')];
  const nnCells = [...root.querySelectorAll('#d-nn .nn')];

  function show(k) {
    sel = k;
    const f = frames[k];
    if (!f) return;
    putImage(xtCv, f.x); putImage(guessCv, f.guess);
    slider.max = frames.length - 1; slider.value = k;
    tval.textContent = f.t.toFixed(2);
    const near = Math.round(f.t * 10);
    cells.forEach((c, i) => c.classList.toggle('sel', i === near && Math.abs(f.t - i / 10) < 0.5 / N + 1e-6));
    $('#d-reroll').textContent = `Re-roll from t = ${f.t.toFixed(2)}`;
    $('#d-reroll').disabled = f.t >= 1;
  }

  function drawStrip() {
    cells.forEach((c, i) => {
      const k = Math.round(i / 10 * N), f = frames[k];
      const cv = c.querySelector('canvas');
      if (f) putImage(cv, f.x); else cv.getContext('2d').clearRect(0, 0, cv.width, cv.height);
    });
  }

  async function develop(newSeed = true) {
    const id = ++job;
    if (newSeed) seed = (Math.random() * 1e6) | 0;
    $('#d-seed').textContent = `noise #${seed}`;
    const x = gaussFill(new Float32Array(D), mulberry32(seed));
    frames = [];
    const h = 1 / N, minMs = 2400 / N;
    for (let i = 0; i < N; i++) {
      const t0 = performance.now(), t = i * h;
      const u = await net.forward(x, new Float32Array([t]), 1);
      if (id !== job) return;
      const guess = new Float32Array(D);
      for (let j = 0; j < D; j++) guess[j] = x[j] + (1 - t) * u[j];
      frames.push({ t, x: x.slice(), guess });
      show(i); drawStrip();
      step(x, u, t, h, sigma, 1);
      await wait(Math.max(0, minMs - (performance.now() - t0)));
      if (id !== job) return;
    }
    frames.push({ t: 1, x: x.slice(), guess: x.slice() });
    show(N); drawStrip();
    nearest(x);
  }

  async function reroll(k) {
    const id = ++rerollJob, f = frames[k];
    if (!f || f.t >= 1) return;
    const x = new Float32Array(N_REROLL * D);
    for (let b = 0; b < N_REROLL; b++) x.set(f.x, b * D);
    const n = N - k, h = (1 - f.t) / n;
    $('#d-sheet-cap').textContent = `Nine continuations of the same image from t = ${f.t.toFixed(2)}, each with its own noise (σ = ${SIGMA_REROLL}).`;
    for (let i = 0; i < n; i++) {
      const t = f.t + i * h;
      const u = await net.forward(x, new Float32Array(N_REROLL).fill(t), N_REROLL);
      if (id !== rerollJob) return;
      step(x, u, t, h, SIGMA_REROLL, N_REROLL);
      if (i % 2 === 1 || i === n - 1) { sheet.forEach((cv, b) => putImage(cv, x, b * D)); await frameWait(); }
    }
  }

  // ---------- nearest training photos ----------
  let atlas = null;
  async function loadAtlas() {
    const meta = await fetch('train_atlas.json').then(r => r.json());
    const img = new Image();
    img.src = 'train_atlas.jpg';
    await img.decode();
    const c = document.createElement('canvas');
    c.width = img.width; c.height = img.height;
    const g = c.getContext('2d', { willReadFrequently: true });
    g.drawImage(img, 0, 0);
    const px = g.getImageData(0, 0, img.width, img.height).data;
    const T = new Float32Array(meta.n * D);
    for (let k = 0; k < meta.n; k++) {
      const r0 = Math.floor(k / meta.cols) * S, c0 = (k % meta.cols) * S;
      for (let y = 0; y < S; y++)
        for (let xx = 0; xx < S; xx++) {
          const p = 4 * ((r0 + y) * img.width + c0 + xx);
          for (let ch = 0; ch < 3; ch++) T[k * D + ch * S * S + y * S + xx] = px[p + ch] / 127.5 - 1;
        }
    }
    atlas = { T, img, meta };
  }
  const atlasReady = loadAtlas().catch(e => console.warn('atlas', e));

  async function nearest(x) {
    await atlasReady;
    if (!atlas) return;
    const { T, meta } = atlas, best = [];
    for (let k = 0; k < meta.n; k++) {
      let a = 0, b = 0;
      const o = k * D;
      for (let ch = 0; ch < 3; ch++)
        for (let y = 0; y < S; y++) {
          const r = ch * S * S + y * S;
          for (let xx = 0; xx < S; xx++) {
            const v = x[r + xx], d1 = v - T[o + r + xx], d2 = v - T[o + r + S - 1 - xx];
            a += d1 * d1; b += d2 * d2;
          }
        }
      best.push([Math.sqrt(Math.min(a, b) / D), k, b < a]);
    }
    best.sort((p, q) => p[0] - q[0]);
    nnCells.forEach((cell, i) => {
      const [d, k, flip] = best[i], cv = cell.querySelector('canvas'), g = cv.getContext('2d');
      cv.width = S; cv.height = S;
      g.save(); if (flip) { g.translate(S, 0); g.scale(-1, 1); }
      g.drawImage(atlas.img, (k % meta.cols) * S, Math.floor(k / meta.cols) * S, S, S, 0, 0, S, S);
      g.restore();
      cell.querySelector('.d').textContent = d.toFixed(3);
    });
  }

  // ---------- controls ----------
  slider.addEventListener('input', () => show(+slider.value));
  cells.forEach((c, i) => c.addEventListener('click', () => { if (frames.length) show(Math.min(frames.length - 1, Math.round(i / 10 * N))); }));
  $('#d-new').addEventListener('click', () => develop(true));
  $('#d-again').addEventListener('click', () => develop(false));
  $('#d-reroll').addEventListener('click', () => reroll(sel));
  root.querySelectorAll('[data-steps]').forEach(b => {
    b.setAttribute('aria-pressed', +b.dataset.steps === N);
    b.addEventListener('click', () => {
      N = +b.dataset.steps;
      root.querySelectorAll('[data-steps]').forEach(x => x.setAttribute('aria-pressed', x === b));
      develop(false);
    });
  });
  const sig = $('#d-sigma'), sigOut = $('#d-sigval');
  sig.addEventListener('input', () => { sigma = +sig.value; sigOut.textContent = sigma === 0 ? '0 (ODE)' : sigma.toFixed(1); });
  sig.addEventListener('change', () => develop(false));

  await develop(true);
  const k0 = Math.round(0.4 * N);
  show(k0);
  reroll(k0);
}
