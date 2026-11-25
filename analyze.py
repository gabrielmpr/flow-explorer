"""Everything measured on the trained model. Writes docs/data.json (numbers for the page's charts),
docs/img/*.png (image grids, also used by the README) and docs/train_atlas.jpg (the training photos, for
nearest-neighbor search in the browser).

python analyze.py            # ~20 min on an M2
python analyze.py --quick    # small sample sizes, for testing
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import UNet, sample, predicted_endpoint

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
args = ap.parse_args()
Q = args.quick

here = os.path.dirname(os.path.abspath(__file__))
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
os.makedirs(os.path.join(here, 'docs', 'img'), exist_ok=True)
S = 32
SIGMA_BRANCH = 1.0   # noise level of the re-rolls
BASE_STEPS = 100     # Euler steps for a full 0 -> 1 run


def load(split):
    paths = sorted(glob.glob(os.path.join(here, 'afhq', split, 'dog', '*')))
    imgs = [np.array(Image.open(p).convert('RGB').resize((S, S), Image.BICUBIC)) for p in paths]
    return torch.tensor(np.stack(imgs)).permute(0, 3, 1, 2).float() / 127.5 - 1


# same split as train.py: seed 0, then the permutation is the first random call
torch.manual_seed(0)
data = load('train')
perm = torch.randperm(len(data))
n = int(len(data) * 0.9)
train = data[perm[:n]]
held = load('val')                      # AFHQ's own validation dogs: never seen in training
print(f'{len(train)} training photos, {len(held)} held-out photos')

ck = torch.load(os.path.join(here, 'ckpt.pt'), map_location='cpu')
net = UNet(ck['n_ch']).to(dev)
net.load_state_dict(ck['model'])
net.eval()
out = {'step': ck['step'], 'sigma_branch': SIGMA_BRANCH, 'base_steps': BASE_STEPS}
gen = lambda seed: torch.Generator(device='cpu').manual_seed(seed)
noise = lambda k, seed: torch.randn(k, 3, S, S, generator=gen(seed)).to(dev)


def rms(a, b):  # per-image root mean square difference, in [-1, 1] pixel units
    return ((a - b) ** 2).flatten(1).mean(1).sqrt()


def to_u8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def grid(rows, pad=2, scale=3):
    """rows: list of lists of (3,S,S) tensors (or None for a gap) -> PIL image, transparent between cells"""
    R, C = len(rows), max(len(r) for r in rows)
    canvas = np.zeros((R * (S + pad) - pad, C * (S + pad) - pad, 4), np.uint8)
    for i, r in enumerate(rows):
        for j, im in enumerate(r):
            if im is not None:
                canvas[i * (S + pad): i * (S + pad) + S, j * (S + pad): j * (S + pad) + S, :3] = to_u8(im[None])[0]
                canvas[i * (S + pad): i * (S + pad) + S, j * (S + pad): j * (S + pad) + S, 3] = 255
    img = Image.fromarray(canvas)
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


@torch.no_grad()
def batched(fn, x, bs=128):
    return torch.cat([fn(x[i:i + bs]) for i in range(0, len(x), bs)])


# ---------- 1. at t = 0 the model's guess is the average dog ----------
mean_img = train.mean(0, keepdim=True).to(dev)
x0 = noise(256, 1)
with torch.no_grad():
    t0 = torch.zeros(len(x0), 1, device=dev)
    guess0 = predicted_endpoint(net(x0, t0), x0, t0)
out['t0_guess'] = {
    'rms_to_mean': float(rms(guess0, mean_img).mean()),
    'rms_photo_to_mean': float(rms(train[:1000].to(dev), mean_img).mean()),
}
print('t=0 guess vs mean image', out['t0_guess'])


# ---------- 2. when does the dog decide? branch an ODE run at time t, re-roll the rest with noise ----------
def bands(x):
    """Split images into three spatial scales that add back up to x."""
    c8 = F.interpolate(F.avg_pool2d(x, 8), scale_factor=8)            # 4x4 blocks: overall color and light
    c2 = F.interpolate(F.avg_pool2d(x, 2), scale_factor=2)            # 16x16: shapes, where the face sits
    return {'color': c8, 'layout': c2 - c8, 'detail': x - c2}


ts = [round(0.1 * i, 1) for i in range(10)]
n_base, n_branch = (12, 4) if Q else (48, 8)
xb = noise(n_base, 2)
_, path = sample(net, xb, BASE_STEPS, return_path=True)                 # the shared ODE run
spread = {k: [] for k in ['color', 'layout', 'detail', 'all']}
example_rows = []
for t in ts:
    k = round(t * BASE_STEPS)
    xt = path[k].repeat_interleave(n_branch, 0)                         # (n_base * n_branch, 3, S, S)
    steps = BASE_STEPS - k
    g = torch.Generator(device=dev).manual_seed(100 + k)
    ends = batched(lambda z: sample(net, z, steps, sigma=SIGMA_BRANCH, t0=t, gen=g), xt, 192)
    e = ends.view(n_base, n_branch, 3, S, S)
    parts = {**bands(ends), 'all': ends}
    for name, v in parts.items():
        v = v.view(n_base, n_branch, -1)
        spread[name].append(float(v.var(1, unbiased=True).mean()))   # variance across re-rolls, averaged
    if k % 20 == 0:                                                     # t = 0, 0.2, ..., 0.8 for the figure
        example_rows.append([e[0, j] for j in range(min(n_branch, 8))])
    print(f't={t}: spread', {kk: round(vv[-1], 4) for kk, vv in spread.items()})

# variance between independent samples, for scale (the re-rolls at t = 0 should match it)
indep = batched(lambda z: sample(net, z, BASE_STEPS), noise(n_base * n_branch, 3), 192)
ref = {**bands(indep), 'all': indep}
ref = {k: float(v.flatten(1).var(0, unbiased=True).mean()) for k, v in ref.items()}
frac = {k: [v / spread[k][0] for v in spread[k]] for k in spread}


def crossing(fs, level=0.5):
    for i in range(1, len(fs)):
        if fs[i] < level <= fs[i - 1]:
            return ts[i - 1] + (fs[i - 1] - level) / (fs[i - 1] - fs[i]) * (ts[i] - ts[i - 1])
    return None


out['commit'] = {'t': ts, 'spread': spread, 'frac': frac, 'independent': ref,
                 't_half': {k: crossing(frac[k]) for k in frac}, 'n_base': n_base, 'n_branch': n_branch}
print('half-decided at', out['commit']['t_half'])
grid(example_rows).save(os.path.join(here, 'docs', 'img', 'rerolls.png'))

# the model's guess of the final image along one ODE run
guess_rows = []
for s in range(4):
    row = []
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        k = round(t * BASE_STEPS)
        x = path[k][s:s + 1]
        tt = torch.full((1, 1), t, device=dev)
        with torch.no_grad():
            row.append(predicted_endpoint(net(x, tt), x, tt)[0])
    row.append(path[-1][s])
    guess_rows.append(row)
grid(guess_rows).save(os.path.join(here, 'docs', 'img', 'guesses.png'))


# ---------- 3. is it copying? nearest training photo for generated vs held-out photos ----------
train_d = train.to(dev)
train_all = torch.cat([train_d, train_d.flip(-1)])                     # training saw both orientations


@torch.no_grad()
def nearest(q, k=3):
    d = torch.cdist(q.flatten(1), train_all.flatten(1)) / math.sqrt(3 * S * S)   # rms distance
    v, i = d.topk(k, largest=False)
    return v.cpu(), i.cpu()


n_gen = 200 if Q else 1000
gens = batched(lambda z: sample(net, z, BASE_STEPS), noise(n_gen, 4), 200)
dg, ig = zip(*[nearest(gens[i:i + 100]) for i in range(0, n_gen, 100)])
dg, ig = torch.cat(dg), torch.cat(ig)
dh, _ = zip(*[nearest(held[i:i + 100].to(dev)) for i in range(0, len(held), 100)])
dh = torch.cat(dh)
bins = np.linspace(0, 0.8, 81)
out['nearest'] = {
    'bins': bins.tolist(),
    'generated': np.histogram(dg[:, 0].numpy(), bins)[0].tolist(),
    'held_out': np.histogram(dh[:, 0].numpy(), bins)[0].tolist(),
    'median_generated': float(dg[:, 0].median()), 'median_held_out': float(dh[:, 0].median()),
    'min_generated': float(dg[:, 0].min()), 'min_held_out': float(dh[:, 0].min()),
    'frac_generated_below_held_min': float((dg[:, 0] < dh[:, 0].min()).float().mean()),
    'n_generated': n_gen, 'n_held_out': len(held),
}
print('nearest', {k: v for k, v in out['nearest'].items() if not isinstance(v, list)})
order = dg[:, 0].argsort()
pick = [order[0].item(), order[len(order) // 4].item(), order[len(order) // 2].item(), order[-1].item()]
nn_rows = [[gens[i]] + [None] + [train_all[j] for j in ig[i]] for i in pick]
grid(nn_rows).save(os.path.join(here, 'docs', 'img', 'nearest.png'))
out['nearest']['examples'] = [[float(v) for v in dg[i]] for i in pick]


# ---------- 4. how many steps? distance to a 500-step reference, and how straight the paths are ----------
n_s = 64 if Q else 256
xs = noise(n_s, 5)
# 500-step reference, summing the path length as it goes (storing the whole path does not fit in 8 GB)
refx, travelled = xs.clone(), torch.zeros(n_s, device=dev)
with torch.no_grad():
    for i in range(500):
        u = net(refx, torch.full((n_s, 1), i / 500, device=dev))
        travelled += (u / 500).flatten(1).norm(dim=1)
        refx = refx + u / 500
steps_list = [1, 2, 3, 5, 10, 20, 50, 100, 200]
dist = []
strip = []
for N in steps_list:
    xN = sample(net, xs, N)
    dist.append(float(rms(xN, refx).mean()))
    strip.append(xN[0])
strip.append(refx[0])
# straightness: straight-line distance from start to end over the length of the path actually travelled
direct = (refx - xs).flatten(1).norm(dim=1)
out['steps'] = {'n': steps_list, 'rms_to_ref': dist, 'ref_steps': 500,
                'straightness': float((direct / travelled).mean()),
                'straightness_min': float((direct / travelled).min()),
                'photo_spread': ref['all'] ** 0.5}
print('steps', out['steps'])
strip_rows = [strip]
for s in range(1, 4):
    strip_rows.append([sample(net, xs[s:s + 1], N)[0] for N in steps_list] + [refx[s]])
grid(strip_rows).save(os.path.join(here, 'docs', 'img', 'steps.png'))

# ---------- training curve ----------
log_path = os.path.join(here, 'train_log.json')
out['train_log'] = json.load(open(log_path)) if os.path.exists(log_path) else []

json.dump(out, open(os.path.join(here, 'docs', 'data.json'), 'w'), indent=1)

# ---------- training photos as one JPEG, for nearest-neighbor search in the browser ----------
cols = 64
rows_n = math.ceil(len(train) / cols)
atlas = np.zeros((rows_n * S, cols * S, 3), np.uint8)
u8 = to_u8(train)
for i in range(len(train)):
    r, c = divmod(i, cols)
    atlas[r * S:(r + 1) * S, c * S:(c + 1) * S] = u8[i]
Image.fromarray(atlas).save(os.path.join(here, 'docs', 'train_atlas.jpg'), quality=95)
json.dump({'n': len(train), 'cols': cols}, open(os.path.join(here, 'docs', 'train_atlas.json'), 'w'))
print('wrote docs/data.json, docs/img/*.png, docs/train_atlas.jpg')
