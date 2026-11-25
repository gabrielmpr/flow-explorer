// The 2-D playground: paint a distribution, train a flow to it in a worker, watch particles follow the field.
import { Field, gauss } from './mlp2d.js';
import { css, setupCanvas } from './draw.js';

const R = 3.2;                   // the canvas shows [-R, R] x [-R, R]
const STEPS = 2000;
const SNAPS = [0, 50, 200, 500, 1000, 2000];
const N_PART = 1000, N_PATH = 120, N_SNAP = 900;

// ---------- presets: each returns a flat [x0, y0, x1, y1, ...] array ----------
function pts(n, f) { const a = new Float32Array(2 * n); for (let i = 0; i < n; i++) { const [x, y] = f(i); a[2 * i] = x; a[2 * i + 1] = y; } return a; }
const PRESETS = {
  spiral: () => pts(1500, () => { const r = Math.sqrt(Math.random()) * 2.7, a = 2.8 * r + Math.PI * (Math.random() < 0.5); return [r * Math.cos(a), r * Math.sin(a)]; }),
  moons: () => pts(1500, i => { const a = Math.PI * Math.random(); return i % 2 ? [1.6 * Math.cos(a) - 0.8, 1.6 * Math.sin(a) - 0.4] : [0.8 - 1.6 * Math.cos(a), 0.4 - 1.6 * Math.sin(a)]; }),
  rings: () => pts(1500, i => { const a = 2 * Math.PI * Math.random(), r = i % 3 ? 2.5 : 1.1; return [r * Math.cos(a), r * Math.sin(a)]; }),
  checkers: () => pts(1500, () => { for (;;) { const x = (Math.random() * 4 - 2) * 1.25, y = (Math.random() * 4 - 2) * 1.25; if ((Math.floor(x / 1.25 + 2) + Math.floor(y / 1.25 + 2)) % 2 === 0) return [x, y]; } }),
  face: () => pts(1500, i => {
    const a = 2 * Math.PI * Math.random();
    if (i % 5 < 2) return [2.6 * Math.cos(a), 2.6 * Math.sin(a)];                       // outline
    if (i % 5 === 2) return [-0.95 + 0.25 * Math.cos(a), 0.9 + 0.35 * Math.sin(a)];     // eyes
    if (i % 5 === 3) return [0.95 + 0.25 * Math.cos(a), 0.9 + 0.35 * Math.sin(a)];
    const b = Math.PI * (1.15 + 0.7 * Math.random()); return [1.5 * Math.cos(b), 0.1 + 1.5 * Math.sin(b)]; // mouth
  }),
};

export function initPlayground(root) {
  const $ = s => root.querySelector(s);
  const cv = $('#p2-canvas'), lossCv = $('#p2-loss'), tSlider = $('#p2-t'), tOut = $('#p2-tval'), status = $('#p2-status');
  const snapCvs = [...root.querySelectorAll('.snap canvas')];
  let data = PRESETS.spiral(), init = null, mode = 'data';
  let params = null, lossLog = [], trainStart = 0, lastStep = 0;
  const field = new Field(), snapField = new Field(), u = new Float32Array(2);
  const worker = new Worker(new URL('./worker2d.js', import.meta.url), { type: 'module' });

  // fixed starting points, so the picture changes only because the field does
  let starts = null, snapStarts = null;
  const sampleInit = n => {
    const a = new Float32Array(2 * n);
    for (let i = 0; i < n; i++) {
      if (init && init.length >= 2) { const j = (Math.random() * (init.length / 2)) | 0; a[2 * i] = init[2 * j] + 0.04 * gauss(); a[2 * i + 1] = init[2 * j + 1] + 0.04 * gauss(); }
      else { a[2 * i] = gauss(); a[2 * i + 1] = gauss(); }
    }
    return a;
  };
  const resetStarts = () => { starts = sampleInit(N_PART); snapStarts = sampleInit(N_SNAP); };

  function train() {
    if (data.length < 20) { status.textContent = 'Paint some points first.'; params = null; return; }
    resetStarts();
    lossLog = []; trainStart = performance.now();
    for (const c of snapCvs) c._img = null;
    worker.postMessage({ data, init: init && init.length >= 20 ? init : null, steps: STEPS, snaps: SNAPS });
  }

  worker.onmessage = e => {
    const m = e.data;
    params = m.params; field.set(params); lastStep = m.step;
    if (m.loss !== null) lossLog.push([m.step, m.loss]);
    if (m.snap) drawSnap(SNAPS.indexOf(m.step), m.params);
    status.textContent = m.done
      ? `Trained: ${STEPS.toLocaleString()} steps in ${((performance.now() - trainStart) / 1000).toFixed(1)} s. Final loss ${m.loss.toFixed(3)}.`
      : `Training: step ${m.step.toLocaleString()} of ${STEPS.toLocaleString()}`;
    pathsDirty = true;
    drawLoss();
  };

  // ---------- animation state ----------
  let t = 0, hold = 0, playing = true, cur = null, pathsDirty = true, paths = null, pathsAt = 0;
  const advance = (x, y, t0, t1, n) => {
    const h = (t1 - t0) / n;
    for (let s = 0; s < n; s++) { field.eval(x, y, t0 + s * h, u); x += h * u[0]; y += h * u[1]; }
    return [x, y];
  };
  const restart = () => { cur = starts.slice(); t = 0; hold = 0; };
  const jumpTo = tt => {
    cur = starts.slice();
    const n = Math.max(1, Math.round(tt * 40));
    if (params) for (let i = 0; i < N_PART; i++) { const [x, y] = advance(cur[2 * i], cur[2 * i + 1], 0, tt, n); cur[2 * i] = x; cur[2 * i + 1] = y; }
    t = tt;
  };
  const computePaths = () => {
    paths = [];
    for (let i = 0; i < N_PATH; i++) {
      let x = starts[2 * i], y = starts[2 * i + 1];
      const p = [x, y];
      for (let s = 0; s < 40; s++) { field.eval(x, y, s / 40, u); x += u[0] / 40; y += u[1] / 40; p.push(x, y); }
      paths.push(p);
    }
    pathsDirty = false;
  };

  // ---------- drawing ----------
  let ctx, W;
  const toPx = v => (v + R) / (2 * R) * W;
  const toPy = v => (R - v) / (2 * R) * W;
  const resize = () => { ({ ctx, w: W } = setupCanvas(cv)); };

  function frame() {
    if (!cur) restart();
    if (playing && params) {
      if (t < 1) {
        const t1 = Math.min(1, t + 1 / 150);
        for (let i = 0; i < N_PART; i++) { const [x, y] = advance(cur[2 * i], cur[2 * i + 1], t, t1, 1); cur[2 * i] = x; cur[2 * i + 1] = y; }
        t = t1;
      } else if (++hold > 70) restart();
      tSlider.value = t; tOut.textContent = t.toFixed(2);
    }
    const c = css();
    ctx.clearRect(0, 0, W, W);
    // axes cross, very light
    ctx.strokeStyle = c.rule; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(toPx(0), 0); ctx.lineTo(toPx(0), W); ctx.moveTo(0, toPy(0)); ctx.lineTo(W, toPy(0)); ctx.stroke();
    // target and (if painted) source
    const dot = (arr, color, r) => { ctx.fillStyle = color; for (let i = 0; i < arr.length; i += 2) ctx.fillRect(toPx(arr[i]) - r / 2, toPy(arr[i + 1]) - r / 2, r, r); };
    ctx.globalAlpha = 0.35; dot(data, c.ink2, 2); if (init) dot(init, c.gold, 2); ctx.globalAlpha = 1;
    if (params && $('#p2-field').checked) drawField(c);
    if (params && $('#p2-paths').checked) {
      if (pathsDirty && (performance.now() - pathsAt > 250 || !paths)) { computePaths(); pathsAt = performance.now(); }
      ctx.lineWidth = 1; ctx.globalAlpha = 0.5;
      for (const p of paths) {
        ctx.strokeStyle = p[0] < 0 ? c.mark : c.blue;
        ctx.beginPath(); ctx.moveTo(toPx(p[0]), toPy(p[1]));
        for (let k = 2; k < p.length; k += 2) ctx.lineTo(toPx(p[k]), toPy(p[k + 1]));
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    if (params) {
      for (let i = 0; i < N_PART; i++) {
        ctx.fillStyle = starts[2 * i] < 0 ? c.mark : c.blue;
        ctx.fillRect(toPx(cur[2 * i]) - 1.2, toPy(cur[2 * i + 1]) - 1.2, 2.4, 2.4);
      }
    }
    requestAnimationFrame(frame);
  }

  function drawField(c) {
    const n = 17, step = 2 * R / n;
    ctx.strokeStyle = c.ink; ctx.globalAlpha = 0.35; ctx.lineWidth = 1;
    ctx.beginPath();
    const tt = Math.min(t, 0.99);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        const x = -R + (i + 0.5) * step, y = -R + (j + 0.5) * step;
        field.eval(x, y, tt, u);
        const m = Math.hypot(u[0], u[1]) || 1e-9, len = Math.min(m, 2.5) / 2.5 * step * 0.9;
        const dx = u[0] / m * len, dy = u[1] / m * len;
        const px = toPx(x), py = toPy(y), ex = toPx(x + dx), ey = toPy(y + dy);
        ctx.moveTo(px, py); ctx.lineTo(ex, ey);
        const a = Math.atan2(ey - py, ex - px), hs = Math.min(4, len / step * 9);
        ctx.moveTo(ex, ey); ctx.lineTo(ex - hs * Math.cos(a - 0.5), ey - hs * Math.sin(a - 0.5));
        ctx.moveTo(ex, ey); ctx.lineTo(ex - hs * Math.cos(a + 0.5), ey - hs * Math.sin(a + 0.5));
      }
    ctx.stroke(); ctx.globalAlpha = 1;
  }

  function drawSnap(k, p) {
    const c2 = snapCvs[k];
    if (!c2) return;
    snapField.set(p);
    const ends = new Float32Array(2 * N_SNAP);
    for (let i = 0; i < N_SNAP; i++) {
      let x = snapStarts[2 * i], y = snapStarts[2 * i + 1];
      for (let s = 0; s < 30; s++) { snapField.eval(x, y, s / 30, u); x += u[0] / 30; y += u[1] / 30; }
      ends[2 * i] = x; ends[2 * i + 1] = y;
    }
    c2._img = { ends, starts: snapStarts };
    paintSnap(c2);
  }
  function paintSnap(c2) {
    const { ctx: g, w } = setupCanvas(c2), c = css();
    g.clearRect(0, 0, w, w);
    const sx = v => (v + R) / (2 * R) * w, sy = v => (R - v) / (2 * R) * w;
    g.fillStyle = c.ink2; g.globalAlpha = 0.25;
    for (let i = 0; i < data.length; i += 4) g.fillRect(sx(data[i]) - 0.6, sy(data[i + 1]) - 0.6, 1.2, 1.2);
    g.globalAlpha = 1;
    if (!c2._img) return;
    const { ends, starts: st } = c2._img;
    for (let i = 0; i < ends.length; i += 2) { g.fillStyle = st[i] < 0 ? c.mark : c.blue; g.fillRect(sx(ends[i]) - 0.9, sy(ends[i + 1]) - 0.9, 1.8, 1.8); }
  }

  function drawLoss() {
    const { ctx: g, w, h } = setupCanvas(lossCv), c = css();
    g.clearRect(0, 0, w, h);
    const padL = 34, padB = 18, pw = w - padL - 6, ph = h - padB - 6;
    g.font = '11px "IBM Plex Mono", monospace'; g.fillStyle = c.ink2; g.strokeStyle = c.rule; g.lineWidth = 1;
    g.beginPath(); g.moveTo(padL, 6); g.lineTo(padL, 6 + ph); g.lineTo(padL + pw, 6 + ph); g.stroke();
    g.textAlign = 'right'; g.textBaseline = 'middle';
    if (lossLog.length < 2) return;
    const ys = lossLog.map(d => d[1]), lo = Math.min(...ys), hi = Math.max(...ys.slice(Math.min(3, ys.length - 1)));
    const y0 = Math.max(0, lo - 0.1 * (hi - lo)), y1 = hi + 0.05 * (hi - lo);
    const Y = v => 6 + ph * (1 - (v - y0) / (y1 - y0)), X = s => padL + pw * s / STEPS;
    g.fillText(y1.toFixed(2), padL - 4, 10); g.fillText(y0.toFixed(2), padL - 4, 6 + ph - 2);
    g.textAlign = 'center'; g.textBaseline = 'top';
    g.fillText('0', padL, 8 + ph); g.fillText(STEPS.toLocaleString() + ' steps', padL + pw - 30, 8 + ph);
    g.strokeStyle = c.ink; g.lineWidth = 1.5; g.beginPath();
    lossLog.forEach(([s, v], i) => (i ? g.lineTo : g.moveTo).call(g, X(s), Y(Math.min(v, y1))));
    g.stroke();
    for (const s of SNAPS.slice(1)) if (s <= lastStep) { g.fillStyle = c.mark; g.fillRect(X(s) - 1, 6 + ph - 5, 2, 5); }
  }

  // ---------- painting ----------
  let painting = false, dirty = false;
  const at = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) / r.width * 2 * R - R, R - (e.clientY - r.top) / r.height * 2 * R]; };
  const spray = e => {
    const [x, y] = at(e), add = new Float32Array(2 * 8);
    for (let i = 0; i < 8; i++) { add[2 * i] = x + 0.12 * gauss(); add[2 * i + 1] = y + 0.12 * gauss(); }
    const cap = a => (a.length > 8000 ? a.subarray(a.length - 8000) : a);
    if (mode === 'data') data = cap(concat(data, add)); else init = cap(concat(init || new Float32Array(0), add));
    dirty = true;
  };
  cv.addEventListener('pointerdown', e => { painting = true; cv.setPointerCapture(e.pointerId); spray(e); });
  cv.addEventListener('pointermove', e => { if (painting) spray(e); });
  const end = () => { if (!painting) return; painting = false; if (dirty) { dirty = false; train(); } };
  cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', end);

  // ---------- controls ----------
  root.querySelectorAll('[data-mode]').forEach(b => b.addEventListener('click', () => {
    mode = b.dataset.mode;
    root.querySelectorAll('[data-mode]').forEach(x => x.setAttribute('aria-pressed', x === b));
  }));
  root.querySelectorAll('[data-preset]').forEach(b => b.addEventListener('click', () => { data = PRESETS[b.dataset.preset](); train(); }));
  $('#p2-clear').addEventListener('click', () => {
    if (mode === 'data') { data = new Float32Array(0); params = null; status.textContent = 'Paint some points first.'; }
    else { init = null; train(); }
  });
  $('#p2-gauss').addEventListener('click', () => { init = null; train(); });
  $('#p2-retrain').addEventListener('click', train);
  tSlider.addEventListener('input', () => { playing = false; $('#p2-play').textContent = 'Play'; jumpTo(+tSlider.value); tOut.textContent = (+tSlider.value).toFixed(2); });
  $('#p2-play').addEventListener('click', e => {
    playing = !playing; e.target.textContent = playing ? 'Pause' : 'Play';
    if (playing && t >= 1) restart();
  });

  const redrawStatic = () => { resize(); drawLoss(); snapCvs.forEach(paintSnap); };
  new ResizeObserver(redrawStatic).observe(cv);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redrawStatic);
  resize();
  resetStarts();
  train();
  requestAnimationFrame(frame);
}

function concat(a, b) { const c = new Float32Array(a.length + b.length); c.set(a); c.set(b, a.length); return c; }
