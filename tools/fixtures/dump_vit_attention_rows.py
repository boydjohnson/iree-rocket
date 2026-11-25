#!/usr/bin/env python3
"""Regenerates iree-rocket-hal/tests/fixtures/vit_b_16_attention_rows.f32.

64 rows of real pre-softmax attention logits, each already reduced by its
own row maximum -- the input SOFTMAX.md 3.3's (input_scale, slope) pair is
tuned against, and the fixture lut_exp_envelope_hw.rs measures total
variation over.

torchvision's ViT-B/16 at IMAGENET1K_V1 on one image. The scores are
recomputed from each encoder block's `in_proj_weight` rather than read from
`output_attentions`, which returns probabilities and not logits: hook
`ln_1`'s output to get the attention input, split the fused QKV projection,
and take `q @ k.T / sqrt(head_dim)`. Rows are sampled every
(12 layers * 12 heads * 197 rows)/64'th so the set spans every layer.

    python3 tools/fixtures/dump_vit_attention_rows.py IMAGE OUT.f32

Needs torch, torchvision and pillow, and the ViT-B/16 weights in the
torch hub cache; it makes no network call of its own. The committed fixture
was built from `mlir-spike/Unicorn.jpg`, the image that tree already used
for ViT, and its distribution is in the test's own doc comment: median
-3.44, 28% of entries past the table's -5 domain edge.
"""
import math, sys
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).eval()
prep = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
image = prep(Image.open(sys.argv[1]).convert("RGB")).unsqueeze(0)

pre = {}
hooks = []

def grab(index):
    def hook(_m, _i, out):
        pre[index] = out.detach()
    return hook

for index, block in enumerate(model.encoder.layers):
    hooks.append(block.ln_1.register_forward_hook(grab(index)))
with torch.no_grad():
    model(image)
for h in hooks:
    h.remove()

heads = 12
rows = []
for index, block in enumerate(model.encoder.layers):
    x = pre[index][0]                       # (tokens, embed)
    embed = x.shape[-1]
    w = block.self_attention.in_proj_weight
    b = block.self_attention.in_proj_bias
    q = x @ w[:embed].T + b[:embed]
    k = x @ w[embed:2 * embed].T + b[embed:2 * embed]
    head_dim = embed // heads
    tokens = x.shape[0]
    q = q.view(tokens, heads, head_dim).permute(1, 0, 2)
    k = k.view(tokens, heads, head_dim).permute(1, 0, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
    rows.append(scores - scores.max(dim=-1, keepdim=True).values)

allr = torch.cat(rows, dim=0).reshape(-1, rows[0].shape[-1])
flat = allr.flatten()
print("tokens %d  rows %d" % (allr.shape[1], allr.shape[0]), file=sys.stderr)
print("x-m: min %.2f  median %.2f  frac<-5 %.4f  frac<-8 %.4f  frac<-16 %.4f"
      % (flat.min(), flat.median(), (flat < -5).float().mean(),
         (flat < -8).float().mean(), (flat < -16).float().mean()),
      file=sys.stderr)
step = allr.shape[0] // 64
sample = allr[::step][:64].contiguous()
sample.detach().numpy().astype("<f4").tofile(sys.argv[2])
print("wrote", sample.shape, file=sys.stderr)
