// The findings charts, drawn from data.json (written by analyze.py). Redrawn on resize and theme change.
import { css, setupCanvas } from './draw.js';

const MONO = '11px "IBM Plex Mono", ui-monospace, monospace';

function frame(cv, { xlab, ylab, padL = 46, padB = 34 }) {
  const { ctx: g, w, h } = setupCanvas(cv), c = css();
  g.clearRect(0, 0, w, h);
  const P = { l: padL, r: w - 14, t: 12, b: h - padB };
  g.strokeStyle = c.ink; g.lineWidth = 1;
  g.beginPath(); g.moveTo(P.l, P.t); g.lineTo(P.l, P.b); g.lineTo(P.r, P.b); g.stroke();
  g.font = MONO; g.fillStyle = c.ink2;
  g.textAlign = 'center'; g.textBaseline = 'alphabetic'; g.fillText(xlab, (P.l + P.r) / 2, h - 4);
  g.save(); g.translate(11, (P.t + P.b) / 2); g.rotate(-Math.PI / 2); g.textBaseline = 'top'; g.fillText(ylab, 0, -4); g.restore();
  return { g, c, P, w, h };
}

function ticks(g, c, P, xs, ys, X, Y, fx, fy) {
  g.font = MONO; g.fillStyle = c.ink2; g.strokeStyle = c.ink;
  g.textAlign = 'center'; g.textBaseline = 'top';
  for (const v of xs) { g.beginPath(); g.moveTo(X(v), P.b); g.lineTo(X(v), P.b + 4); g.stroke(); g.fillText(fx(v), X(v), P.b + 6); }
  g.textAlign = 'right'; g.textBaseline = 'middle';
  for (const v of ys) {
    g.beginPath(); g.moveTo(P.l - 4, Y(v)); g.lineTo(P.l, Y(v)); g.stroke(); g.fillText(fy(v), P.l - 7, Y(v));
    g.save(); g.strokeStyle = c.rule; g.beginPath(); g.moveTo(P.l + 1, Y(v)); g.lineTo(P.r, Y(v)); g.stroke(); g.restore();
  }
}

function label(g, text, x, y, color, align = 'left') {
  g.font = '600 12px "Archivo", sans-serif'; g.fillStyle = color; g.textAlign = align; g.textBaseline = 'middle'; g.fillText(text, x, y);
}

export function drawCommit(cv, d) {
  const { g, c, P } = frame(cv, { xlab: 't where the re-roll starts', ylab: 'variation left' });
  const X = t => P.l + (P.r - P.l - 70) * t / 0.9, Y = v => P.b - (P.b - P.t) * v / 1.05;
  ticks(g, c, P, [0, 0.2, 0.4, 0.6, 0.8], [0, 0.25, 0.5, 0.75, 1], X, Y, v => v.toFixed(1), v => `${Math.round(v * 100)}%`);
  const series = [['color', 'color, lighting', c.gold], ['layout', 'layout', c.mark], ['detail', 'fur, edges', c.blue]];
  // end labels, pushed apart so they don't overlap (lowest first, each at least 14 px above the one below)
  const ly = {}, ends = series.map(([k]) => [k, Y(d.commit.frac[k].at(-1))]).sort((a, b) => b[1] - a[1]);
  let prev = Infinity;
  for (const [k, y] of ends) { ly[k] = Math.min(y, prev - 14); prev = ly[k]; }
  for (const [k, name, col] of series) {
    const f = d.commit.frac[k];
    g.strokeStyle = col; g.lineWidth = 2; g.beginPath();
    f.forEach((v, i) => (i ? g.lineTo : g.moveTo).call(g, X(d.commit.t[i]), Y(v))); g.stroke();
    g.fillStyle = col; f.forEach((v, i) => { g.beginPath(); g.arc(X(d.commit.t[i]), Y(v), 2.5, 0, 7); g.fill(); });
    label(g, name, X(0.9) + 8, ly[k], col);
    const th = d.commit.t_half[k];
    if (th !== null) { g.strokeStyle = col; g.setLineDash([3, 3]); g.lineWidth = 1; g.beginPath(); g.moveTo(X(th), Y(0.5)); g.lineTo(X(th), P.b); g.stroke(); g.setLineDash([]); }
  }
  g.strokeStyle = c.ink2; g.setLineDash([2, 4]); g.lineWidth = 1; g.beginPath(); g.moveTo(P.l, Y(0.5)); g.lineTo(X(0.9), Y(0.5)); g.stroke(); g.setLineDash([]);
}

export function drawNearest(cv, d) {
  const n = d.nearest, { g, c, P } = frame(cv, { xlab: 'distance to the nearest training photo (rms, pixels in [-1, 1])', ylab: 'share of images' });
  const bins = n.bins, norm = a => { const s = a.reduce((p, q) => p + q, 0); return a.map(v => v / s); };
  const A = norm(n.held_out), B = norm(n.generated), top = Math.max(...A, ...B) * 1.1;
  const last = Math.max(...n.held_out.map((v, i) => (v ? i : 0)), ...n.generated.map((v, i) => (v ? i : 0)));
  const lo = bins[0], hi = Math.min(bins[bins.length - 1], Math.ceil(bins[last + 1] * 10 + 0.5) / 10);
  const X = v => P.l + (P.r - P.l) * (v - lo) / (hi - lo), Y = v => P.b - (P.b - P.t) * v / top;
  const yt = []; for (let v = 0; v <= top; v += top > 0.2 ? 0.1 : 0.05) yt.push(v);
  const xt = []; for (let v = 0; v <= hi + 1e-9; v += 0.1) xt.push(v);
  ticks(g, c, P, xt, yt, X, Y, v => v.toFixed(1), v => `${Math.round(v * 100)}%`);
  const hist = (a, col, fill) => {
    g.beginPath(); g.moveTo(X(lo), P.b);
    a.forEach((v, i) => { if (bins[i + 1] <= hi + 1e-9) { g.lineTo(X(bins[i]), Y(v)); g.lineTo(X(bins[i + 1]), Y(v)); } });
    g.lineTo(X(hi), P.b);
    if (fill) { g.fillStyle = col; g.globalAlpha = 0.22; g.fill(); g.globalAlpha = 1; }
    g.strokeStyle = col; g.lineWidth = 1.6; g.stroke();
  };
  hist(A, c.ink2, true); hist(B, c.mark, false);
  const peak = (a) => { let i = a.indexOf(Math.max(...a)); return [X((bins[i] + bins[i + 1]) / 2), Y(a[i])]; };
  const [ax, ay] = peak(A), [bx, by] = peak(B);
  label(g, `held-out photos (${n.n_held_out})`, ax - 10, ay - 12, c.ink2, 'right');
  label(g, `generated (${n.n_generated})`, bx + 10, by - 12, c.mark, 'left');
}

export function drawSteps(cv, d) {
  const s = d.steps, { g, c, P } = frame(cv, { xlab: 'Euler steps (log scale)', ylab: 'distance to the 500-step image' });
  const lx = Math.log10, x0 = lx(1), x1 = lx(500);
  const between = Math.sqrt(2) * s.photo_spread, top = between * 1.12;
  const X = n => P.l + (P.r - P.l) * (lx(n) - x0) / (x1 - x0), Y = v => P.b - (P.b - P.t) * v / top;
  const yt = []; for (let v = 0; v <= top; v += 0.1) yt.push(v);
  ticks(g, c, P, [1, 2, 5, 10, 20, 50, 100, 200, 500], yt, X, Y, v => String(v), v => v.toFixed(1));
  g.strokeStyle = c.ink2; g.setLineDash([4, 4]); g.lineWidth = 1;
  g.beginPath(); g.moveTo(P.l, Y(between)); g.lineTo(P.r, Y(between)); g.stroke(); g.setLineDash([]);
  label(g, 'two different dogs', P.r, Y(between) - 10, c.ink2, 'right');
  g.strokeStyle = c.mark; g.lineWidth = 2; g.beginPath();
  s.n.forEach((n, i) => (i ? g.lineTo : g.moveTo).call(g, X(n), Y(s.rms_to_ref[i]))); g.stroke();
  g.fillStyle = c.mark;
  s.n.forEach((n, i) => { g.beginPath(); g.arc(X(n), Y(s.rms_to_ref[i]), 3, 0, 7); g.fill(); });
}

export function drawTrain(cv, d) {
  const L = d.train_log.filter(r => r.step > 0);
  if (!L.length) return;
  const { g, c, P } = frame(cv, { xlab: 'training step', ylab: 'loss', padL: 46 });
  const maxS = L[L.length - 1].step, ys = L.flatMap(r => [r.train, r.val]);
  const lo = Math.floor(Math.min(...ys) * 100) / 100 - 0.005, hi = Math.max(...ys) + 0.005;
  const X = s => P.l + (P.r - P.l) * s / maxS, Y = v => P.b - (P.b - P.t) * (v - lo) / (hi - lo);
  const step = (hi - lo) > 0.1 ? 0.05 : 0.01, yt = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) yt.push(v);
  const xt = []; for (let s = 0; s <= maxS; s += 5000) xt.push(s);
  ticks(g, c, P, xt, yt, X, Y, v => (v ? `${v / 1000}k` : '0'), v => v.toFixed(2));
  for (const [k, col] of [['train', c.ink2], ['val', c.mark]]) {
    g.strokeStyle = col; g.lineWidth = 1.6; g.beginPath();
    L.forEach((r, i) => (i ? g.lineTo : g.moveTo).call(g, X(r.step), Y(r[k]))); g.stroke();
  }
  const last = L[L.length - 1];
  label(g, 'train', X(last.step) - 4, Y(last.train) - 10, c.ink2, 'right');
  label(g, 'held-out', X(last.step) - 4, Y(last.val) + 12, c.mark, 'right');
}

export async function initCharts() {
  const d = await fetch('data.json').then(r => r.json());
  const all = [['#c-commit', drawCommit], ['#c-nearest', drawNearest], ['#c-steps', drawSteps], ['#c-train', drawTrain]]
    .map(([s, f]) => [document.querySelector(s), f]).filter(([el]) => el);
  const redraw = () => all.forEach(([el, f]) => f(el, d));
  all.forEach(([el]) => new ResizeObserver(redraw).observe(el));
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);
  document.fonts?.ready.then(redraw);
  redraw();
  return d;
}
