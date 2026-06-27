"""Validate line-robustness of the colour head on HELD-OUT data (true labels).

Diagnosis (from real test img 01956): a red/dark interference LINE crossing a
NON-red character makes the colour head output red (false positive), because it
keys on the line's red, not the character's colour. 9/15 current primaries
(v2hs) never trained with line aug.

This checks, on the held-out val split (true colours known, NO test data):
  - CLEAN colour FP/FN: ensure the new colour heads are not worse on clean.
  - LINE-CONTAMINATED colour FP/FN: overlay synthetic red/dark lines and measure
    how often a NON-red char is flipped to "red" (the bug). The fix wins if the
    new (clr) ensemble has far fewer contaminated false-positives while keeping
    clean accuracy.

Caveat: the contamination generator overlaps clr's training aug, so absolute
numbers are optimistic; the decision signal is clr-vs-old under the SAME
contamination (old never saw it) + clean not regressed. Platform is final arbiter.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as VF

import config
from augment import _draw_red_lines, _draw_occlude_lines
from dataset import build_train_dataset, deterministic_split_indices
from predict import load_model

CK = "outputs/checkpoints"
OLD = [f"{CK}/best_v2hs{i}.pt" for i in range(1, 10)] + [f"{CK}/best_phl{i}.pt" for i in range(1, 7)]


@torch.no_grad()
def color_prob(models, images, x_tta=True):
    colorp = None
    shifts = (-4, 0, 4) if x_tta else (0,)
    for sh in shifts:
        sx = images if sh == 0 else VF.affine(images, angle=0, translate=[sh, 0], scale=1.0,
                                              shear=[0, 0], interpolation=InterpolationMode.BILINEAR, fill=[1., 1., 1.])
        for m in models:
            _, kl = m(sx)
            c = F.softmax(kl, -1)
            colorp = c if colorp is None else colorp + c
    return (colorp / colorp.sum(-1, keepdim=True))[..., config.RED_INDEX]


def contaminate(img, seed):
    """Overlay test-like interference lines (distinct params from training:
    thin, dark-red biased) so the check isn't a pure train=test echo."""
    random.seed(seed); torch.manual_seed(seed)
    out = img.clone()
    out = _draw_red_lines(out, n=random.randint(1, 2))       # red lines
    if random.random() < 0.7:
        out = _draw_occlude_lines(out, n=random.randint(1, 2))  # dark/arbitrary lines
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clr", type=Path, nargs="+", required=True, help="new colour-line-robust checkpoints")
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()
    dev = torch.device("cuda")
    base = build_train_dataset(cache_in_ram=True)
    _, vi = deterministic_split_indices(len(base))
    old = [load_model(p, dev) for p in OLD]
    clr = [load_model(p, dev) for p in args.clr]

    clean_imgs, cont_imgs, ktrue = [], [], []
    for n, idx in enumerate(vi):
        img, _, kt = base[idx]
        clean_imgs.append(img); cont_imgs.append(contaminate(img, args.seed + idx)); ktrue.append(kt)
    ktrue = torch.stack(ktrue)
    red_true = (ktrue == config.RED_INDEX)

    def eval_set(imgs, models):
        ps = []
        for i in range(0, len(imgs), 256):
            b = torch.stack(imgs[i:i+256]).to(dev)
            ps.append(color_prob(models, b).cpu())
        p = torch.cat(ps)
        pred = p.ge(args.threshold)
        fp = int((pred & ~red_true).sum())   # non-red called red
        fn = int((~pred & red_true).sum())   # red missed
        return fp, fn

    print(f"held-out val: {len(vi)} imgs, {int(red_true.sum())} red / {int((~red_true).sum())} non-red positions; thr={args.threshold}")
    print(f"{'condition':>22} {'ensemble':>8} {'FP(非红→红)':>12} {'FN(红→漏)':>10}")
    for cond, imgs in (("CLEAN", clean_imgs), ("LINE-CONTAMINATED", cont_imgs)):
        for nm, models in (("old15", old), ("clr", clr)):
            fp, fn = eval_set(imgs, models)
            print(f"{cond:>22} {nm:>8} {fp:>12} {fn:>10}")


if __name__ == "__main__":
    main()
