"""Train u_theta on AFHQ dog faces at 32x32 with the straight-line flow matching loss.

python train.py [--iters 40000] [--n_ch 64]
Writes ckpt.pt (EMA weights) and train_log.json.
"""
import argparse
import copy
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import UNet

p = argparse.ArgumentParser()
p.add_argument('--iters', type=int, default=40000)
p.add_argument('--n_ch', type=int, default=64)
p.add_argument('--batch', type=int, default=64)
p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--time_only', action='store_true')
args = p.parse_args()

img_size, ema_decay, eval_interval = 32, 0.999, 1000
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
here = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)


def load(split_dir):
    paths = sorted(glob.glob(os.path.join(here, 'afhq', split_dir, 'dog', '*')))
    imgs = [np.array(Image.open(q).convert('RGB').resize((img_size, img_size), Image.BICUBIC)) for q in paths]
    return torch.tensor(np.stack(imgs)).permute(0, 3, 1, 2).float() / 127.5 - 1


data = load('train')
data = data[torch.randperm(len(data))]
n = int(len(data) * 0.9)
train_data, val_data = data[:n].to(device), data[n:].to(device)
print(f'{len(data)} images, device = {device}')


def get_batch(split):
    d = train_data if split == 'train' else val_data
    z = d[torch.randint(len(d), (args.batch,), device=device)]
    flip = torch.rand(args.batch, 1, 1, 1, device=device) < 0.5
    return torch.where(flip, z.flip(-1), z)


def loss_fn(net, z):
    t = torch.rand(z.shape[0], 1, 1, 1, device=device)
    eps = torch.randn_like(z)
    x = t * z + (1 - t) * eps
    return F.mse_loss(net(x, t.view(-1, 1)), z - eps)


model = UNet(args.n_ch).to(device)
ema = copy.deepcopy(model).eval()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
print(f'{sum(q.numel() for q in model.parameters()) / 1e6:.2f}M params')


@torch.no_grad()
def estimate(net):
    out = {}
    for split in ['train', 'val']:
        ls = []
        for _ in range(20):
            torch.manual_seed(len(ls))
            ls.append(loss_fn(net, get_batch(split)).item())
        out[split] = float(np.mean(ls))
    return out


log, t_start = [], time.time()
for k in range(args.iters + 1):
    if k % eval_interval == 0 and not args.time_only:
        e = estimate(ema)
        log.append({'step': k, **e, 'sec': time.time() - t_start})
        print(f"step {k}  ema train {e['train']:.4f}  val {e['val']:.4f}  {time.time() - t_start:.0f}s", flush=True)
        torch.save({'model': ema.state_dict(), 'n_ch': args.n_ch, 'step': k}, os.path.join(here, 'ckpt.pt'))
        json.dump(log, open(os.path.join(here, 'train_log.json'), 'w'))
    if k == args.iters:
        break
    torch.manual_seed(10_000 + k)
    loss = loss_fn(model, get_batch('train'))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    with torch.no_grad():
        for pe, q in zip(ema.parameters(), model.parameters()):
            pe.lerp_(q, 1 - ema_decay)
    if args.time_only and k in (20, 120):
        if device == 'mps':
            torch.mps.synchronize()
        if k == 20:
            t20 = time.time()
        else:
            print(f'{(time.time() - t20) / 100 * 1000:.0f} ms/iter')
            break
