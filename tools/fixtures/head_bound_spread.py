"""ROADMAP lever 3: how much slack does a per-head static bound actually need?

Answered, in SOFTMAX.md section 12.1: a median of 78 codes on the typical
row and 495 on the worst, against an accurate exp domain 100 codes wide, so
lever 3 is closed negative. This script is what measured it and is kept so
the number can be rechecked on other models and other images.

    PROJ=~/projects/iree-rocket
    $PROJ/.venv/bin/python tools/fixtures/head_bound_spread.py IMAGE [IMAGE...]

Needs torch, torchvision, pillow and the ViT-B/16 weights in the torch hub
cache; it makes no network call of its own. Pass more than one image to get
the across-input drift, which is the half that decides whether an offline
calibration is possible at all.

Lever 3 replaces softmax's exact per-row maximum with a bound calibrated
offline. The cost is the slack every row pays: bound - that row's own max,
in int8 code units at SOFTMAX.md's input_scale = 1/20, against an exp table
whose accurate domain is 100 codes wide.

Prints, per image and across images, the raw per-head logit geometry that
the committed fixture cannot show -- it is stored already reduced by each
row's own maximum, so its spread is zero by construction.
"""
import math, sys
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

INPUT_SCALE = 1.0 / 20.0
DOMAIN_CODES = 100  # EXP_DOMAIN_EDGE 5.0 / INPUT_SCALE

model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).eval()
prep = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def head_scores(path):
    """Returns a (144, 197, 197) tensor of raw pre-softmax logits."""
    image = prep(Image.open(path).convert("RGB")).unsqueeze(0)
    pre, hooks = {}, []

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
    out = []
    for index, block in enumerate(model.encoder.layers):
        x = pre[index][0]
        embed = x.shape[-1]
        w = block.self_attention.in_proj_weight
        b = block.self_attention.in_proj_bias
        q = x @ w[:embed].T + b[:embed]
        k = x @ w[embed:2 * embed].T + b[embed:2 * embed]
        head_dim = embed // heads
        tokens = x.shape[0]
        q = q.view(tokens, heads, head_dim).permute(1, 0, 2)
        k = k.view(tokens, heads, head_dim).permute(1, 0, 2)
        out.append(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim))
    return torch.cat(out, dim=0)


paths = sys.argv[1:]
per_image = {}
for path in paths:
    scores = head_scores(path)            # (144, 197, 197)
    row_max = scores.max(dim=-1).values   # (144, 197)
    per_image[path] = row_max
    head_hi = row_max.max(dim=-1).values  # (144,)
    head_lo = row_max.min(dim=-1).values
    within = head_hi - head_lo
    name = path.rsplit("/", 1)[-1]
    print(f"\n{name}: 144 heads, 197 rows each")
    print(f"  raw logit range          {scores.min():+.2f} .. {scores.max():+.2f} "
          f"({(scores.max()-scores.min())*20:.0f} codes; int8 holds 255)")
    print(f"  per-head row-max spread  median {within.median():.2f}, "
          f"worst {within.max():.2f} real units "
          f"= median {within.median()*20:.0f}, worst {within.max()*20:.0f} codes")
    frac = (within * 20 > DOMAIN_CODES).float().mean()
    print(f"  heads whose own spread already exceeds the {DOMAIN_CODES}-code domain: "
          f"{frac*100:.0f}%")

if len(paths) > 1:
    print(f"\nacross {len(paths)} images, per head:")
    stacked = torch.stack([per_image[p].max(dim=-1).values for p in paths])  # (img, 144)
    drift = stacked.max(dim=0).values - stacked.min(dim=0).values
    print(f"  the per-head bound itself moves by median {drift.median():.2f}, "
          f"worst {drift.max():.2f} real units "
          f"= median {drift.median()*20:.0f}, worst {drift.max()*20:.0f} codes")
    # A bound calibrated on all images, and the slack it forces on every row.
    bound = stacked.max(dim=0).values                      # (144,)
    worst_slack = torch.zeros(len(bound))
    median_slack = torch.zeros(len(bound))
    for i, path in enumerate(paths):
        s = bound.unsqueeze(-1) - per_image[path]          # (144, 197)
        worst_slack = torch.maximum(worst_slack, s.max(dim=-1).values)
        median_slack = torch.maximum(median_slack, s.median(dim=-1).values)
    print(f"  slack a calibrated bound forces: typical row median "
          f"{median_slack.median()*20:.0f} codes, worst row "
          f"{worst_slack.max()*20:.0f} codes, against a {DOMAIN_CODES}-code domain")
