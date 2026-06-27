"""Option (A): character-localized, hue-based colour recheck to kill neighbour-bleed
false-positive reds.

Mechanism (confirmed on test img 01956): the learned colour head for a slot sees
a region wide enough to include the adjacent RED characters (M at -40px, L at
+40px) and a thin crossing line, so it calls a clearly non-red (green) character
"red". Fix: for each slot the head calls RED, look ONLY at the central +-18px
window (excludes neighbours at +-40px) and compare red ink-mass vs non-red
(green+blue) ink-mass; if the character is clearly non-red dominant, override.

Validated on held-out (true colours, NO test labels): pick the mass-ratio beta so
that ~0 TRUE-RED slots are ever overridden (safety), then apply to test. Platform
is the final arbiter.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as VF
from PIL import Image

import config
from dataset import build_train_dataset, deterministic_split_indices, load_submission_sample, Sample, RedCharDataset
from predict import load_model

CK = "outputs/checkpoints"
PRIM = [f"{CK}/best_v2hs{i}.pt" for i in range(1, 10)] + [f"{CK}/best_phl{i}.pt" for i in range(1, 7)]
CENTERS = [(2 * k + 1) * config.IMAGE_WIDTH // (2 * config.NUM_POSITIONS) for k in range(5)]  # 20,60,100,140,180


def slot_masses(img_hw3, half=18):
    """img: [60,200,3] in [0,1]. Returns per-slot (red_mass, nonred_mass) over
    coloured-ink pixels in the central +-half window (excludes +-40px neighbours)."""
    R, Gc, B = img_hw3[..., 0], img_hw3[..., 1], img_hw3[..., 2]
    sat = img_hw3.max(-1) - img_hw3.min(-1)
    ink = sat > 0.18
    redness = np.clip(R - np.maximum(Gc, B), 0, None)
    nonred = np.clip(Gc - np.maximum(R, B), 0, None) + np.clip(B - np.maximum(R, Gc), 0, None)
    rm, nm = [], []
    for c in CENTERS:
        x0, x1 = max(0, c - half), min(config.IMAGE_WIDTH, c + half)
        m = ink[3:57, x0:x1]
        rm.append(float(redness[3:57, x0:x1][m].sum()))
        nm.append(float(nonred[3:57, x0:x1][m].sum()))
    return np.array(rm), np.array(nm)


@torch.no_grad()
def head_redP(models, loader_imgs, dev):
    out = []
    for i in range(0, len(loader_imgs), 256):
        b = torch.stack(loader_imgs[i:i + 256]).to(dev)
        cp = None
        for sh in (-4, 0, 4):
            sx = b if sh == 0 else VF.affine(b, angle=0, translate=[sh, 0], scale=1.0, shear=[0, 0],
                                             interpolation=InterpolationMode.BILINEAR, fill=[1., 1., 1.])
            for m in models:
                _, kl = m(sx); c = F.softmax(kl, -1)
                cp = c if cp is None else cp + c
        out.append((cp / cp.sum(-1, keepdim=True))[..., config.RED_INDEX].cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", type=int, default=18)
    ap.add_argument("--red-threshold", type=float, default=0.20)
    args = ap.parse_args()
    dev = torch.device("cuda")
    prim = [load_model(p, dev) for p in PRIM]

    # ---------- held-out: safety (never override a TRUE red) ----------
    base = build_train_dataset(cache_in_ram=True)
    _, vi = deterministic_split_indices(len(base))
    imgs, ktrue = [], []
    for idx in vi:
        img, _, kt = base[idx]; imgs.append(img); ktrue.append(kt)
    ktrue = torch.stack(ktrue).numpy()            # [N,5]
    RM = np.zeros((len(vi), 5)); NM = np.zeros((len(vi), 5))
    for j, idx in enumerate(vi):
        a = base[idx][0].permute(1, 2, 0).numpy()
        RM[j], NM[j] = slot_masses(a, args.half)
    redP = head_redP(prim, imgs, dev)
    is_red = ktrue == config.RED_INDEX
    head_red = redP >= args.red_threshold
    print(f"held-out {len(vi)} imgs | true red slots={int(is_red.sum())} nonred={int((~is_red).sum())} | half={args.half}")
    print("override rule: head判红 且 nonred_mass > beta*red_mass  -> 翻成非红")
    print(f"{'beta':>5} {'误翻真红(BAD)':>12} {'翻对真非红假阳':>14} {'净改善估计':>10}")
    for beta in (1.0, 1.5, 2.0, 3.0, 5.0):
        override = head_red & (NM > beta * RM)
        bad = int((override & is_red).sum())          # true-red wrongly flipped (must be ~0)
        good = int((override & ~is_red).sum())         # true-nonred that head FP'd, now fixed
        print(f"{beta:>5.1f} {bad:>12} {good:>14} {good-bad:>10}")
    # how many true-red slots even have nonred>red (intrinsic risk)
    print(f"  (真红槽中 nonred_mass>red_mass 的占比: {float((is_red&(NM>RM)).sum())/max(1,is_red.sum()):.4f})")

    # ---------- test: apply + check the two known cases ----------
    print("\n=== 测试集应用 ===")
    ids = [r.id for r in load_submission_sample().itertuples(index=False)]
    tds = RedCharDataset([Sample(i) for i in ids], config.TEST_IMAGES, is_test=True, cache_in_ram=False)
    timgs = [tds[i][0] for i in range(len(ids))]
    tredP = head_redP(prim, timgs, dev)
    tRM = np.zeros((len(ids), 5)); tNM = np.zeros((len(ids), 5))
    for j, i in enumerate(ids):
        a = np.asarray(Image.open(config.TEST_IMAGES / i).convert("RGB"), dtype=np.float32) / 255.
        tRM[j], tNM[j] = slot_masses(a, args.half)
    thead_red = tredP >= args.red_threshold
    for beta in (1.5, 2.0, 3.0):
        ov = thead_red & (tNM > beta * tRM)
        print(f"  beta={beta}: 翻掉 {int(ov.sum())} 个判红槽, 涉及 {int(ov.any(1).sum())} 张图")
    # specific cases
    itc = config.IDX_TO_CHAR
    for tid, slot in (("01956.png", 2), ("01043.png", 4), ("00634.png", 3), ("03902.png", 1)):
        j = ids.index(tid)
        print(f"  {tid} slot{slot}: P(red)={tredP[j,slot]:.3f} red_mass={tRM[j,slot]:.2f} nonred_mass={tNM[j,slot]:.2f} "
              f"ratio={tNM[j,slot]/max(1e-6,tRM[j,slot]):.2f} -> beta2翻? {'YES' if (thead_red[j,slot] and tNM[j,slot]>2*tRM[j,slot]) else 'no'}")


if __name__ == "__main__":
    main()
