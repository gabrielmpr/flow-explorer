// Small canvas helpers shared by the page's figures.
export function css() {
  const s = getComputedStyle(document.documentElement), v = n => s.getPropertyValue(n).trim();
  return { bg: v('--bg'), sheet: v('--sheet'), ink: v('--ink'), ink2: v('--ink-2'), rule: v('--rule'), mark: v('--mark'), blue: v('--blue'), gold: v('--gold') };
}

// size the backing store to the element's CSS size times the pixel ratio; draw in CSS pixels
export function setupCanvas(cv) {
  const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

// a (3, 32, 32) float image in [-1, 1] -> 32x32 canvas pixels
export function putImage(cv, x, off = 0) {
  if (cv.width !== 32) { cv.width = 32; cv.height = 32; }
  const g = cv.getContext('2d'), im = g.createImageData(32, 32), n = 1024;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) im.data[4 * i + c] = Math.max(0, Math.min(255, Math.round((x[off + c * n + i] + 1) * 127.5)));
    im.data[4 * i + 3] = 255;
  }
  g.putImageData(im, 0, 0);
}
