"""Export the EMA U-Net for the browser.

Writes docs/model.bin (fp16 weights), docs/model.json (tensor table) and docs/parity.json
(inputs + PyTorch outputs, computed with the fp16-rounded weights, for checking unet.js).
"""
import json
import os
import sys

import numpy as np
import torch

from model import UNet

here = os.path.dirname(os.path.abspath(__file__))
ckpt_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, 'ckpt.pt')
ck = torch.load(ckpt_path, map_location='cpu')
model = UNet(ck['n_ch'])
model.load_state_dict(ck['model'])
model.eval()

# round weights to fp16 so PyTorch's parity outputs use exactly what the browser gets
with torch.no_grad():
    for p in model.parameters():
        p.copy_(p.half().float())

table, chunks, offset = [], [], 0
for name, t in model.state_dict().items():
    if name.endswith('freqs'):
        continue  # recomputed in JS
    a = t.detach().numpy().astype(np.float16).ravel()
    table.append({'name': name, 'shape': list(t.shape), 'offset': offset, 'size': a.size})
    chunks.append(a)
    offset += a.size
out = os.path.join(here, 'docs')
np.concatenate(chunks).tofile(os.path.join(out, 'model.bin'))
json.dump({'n_ch': ck['n_ch'], 'step': ck.get('step'), 'tensors': table}, open(os.path.join(out, 'model.json'), 'w'))

g = torch.Generator().manual_seed(0)
x = torch.randn(3, 3, 32, 32, generator=g)
t = torch.tensor([[0.0], [0.37], [0.93]])
with torch.no_grad():
    u = model(x, t)
r = lambda a: [round(float(v), 6) for v in a.ravel()]
json.dump({'x': r(x), 't': r(t), 'u': r(u)}, open(os.path.join(out, 'parity.json'), 'w'))
print(f'{offset / 1e6:.2f}M weights -> {offset * 2 / 1e6:.1f} MB, step {ck.get("step")}')
