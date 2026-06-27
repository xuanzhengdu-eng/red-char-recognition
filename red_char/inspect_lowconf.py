"""Surface the lowest-confidence TEST predictions of the production (98.98) system.

Looks at TEST INPUT images only (no labels) to diagnose where the model is
uncertain on the real test distribution -- the only honest signal left after the
val / 6231 / hard-val proxies all failed to predict the platform. Produces two
annotated montages so we can see whether the real bottleneck is COLOUR
(red/non-red borderline) or CHAR (glyph identity borderline), plus a CSV.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as VF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from dataset import RedCharDataset, Sample, load_submission_sample, decode_prediction
from eval_reranker import selective_rerank
from glyph import glyph_probabilities, load_glyph_model
from predict import load_model

CK = "outputs/checkpoints"
PRIM = [f"{CK}/best_v2hs{i}.pt" for i in range(1, 10)] + [f"{CK}/best_phl{i}.pt" for i in range(1, 7)]
GHL = [f"{CK}/best_ghl{i}.pt" for i in range(1, 4)]
RED_TH = 0.20
PM_MAX, GM_MIN = 0.50, 0.05
IDX2CH = config.IDX_TO_CHAR


@torch.no_grad()
def main() -> None:
    dev = torch.device("cuda")
    samp = load_submission_sample()
    ids = [r.id for r in samp.itertuples(index=False)]
    ds = RedCharDataset([Sample(i) for i in ids], config.TEST_IMAGES, is_test=True, cache_in_ram=False)
    ld = DataLoader(ds, batch_size=200, shuffle=False, num_workers=4)
    prim = [load_model(p, dev) for p in PRIM]
    gly = [load_glyph_model(p, dev) for p in GHL]

    all_char_pred, all_color_p, all_char_top2, all_glyph_top2 = [], [], [], []
    imgs_cpu = []
    for images, _ in ld:
        images = images.to(dev)
        cprob = colorp = None
        for sh in (-4, 0, 4):
            shifted = images if sh == 0 else VF.affine(images, angle=0, translate=[sh, 0], scale=1.0,
                                                        shear=[0, 0], interpolation=InterpolationMode.BILINEAR,
                                                        fill=[1., 1., 1.])
            for m in prim:
                cl, kl = m(shifted)
                cprob = F.softmax(cl, -1) if cprob is None else cprob + F.softmax(cl, -1)
                colorp = F.softmax(kl, -1) if colorp is None else colorp + F.softmax(kl, -1)
        cprob = cprob / cprob.sum(-1, keepdim=True)
        colorp = colorp / colorp.sum(-1, keepdim=True)
        gprob = glyph_probabilities(gly, images)
        char_pred = selective_rerank(cprob, gprob, 3, PM_MAX, GM_MIN)
        # top-2 margins
        c2 = cprob.topk(2, -1).values; g2 = gprob.topk(2, -1).values
        all_char_pred.append(char_pred.cpu())
        all_color_p.append(colorp[..., config.RED_INDEX].cpu())
        all_char_top2.append((c2[..., 0] - c2[..., 1]).cpu())
        all_glyph_top2.append((g2[..., 0] - g2[..., 1]).cpu())
        imgs_cpu.append(images.cpu())
    char_pred = torch.cat(all_char_pred)            # [N,5]
    red_p = torch.cat(all_color_p)                  # [N,5] P(red)
    char_margin = torch.cat(all_char_top2)          # [N,5] primary top1-top2
    glyph_margin = torch.cat(all_glyph_top2)        # [N,5]
    images = torch.cat(imgs_cpu)                    # [N,3,60,200]
    N = char_pred.shape[0]
    red_mask = red_p.ge(RED_TH)                     # predicted red

    # per-image weakness
    color_weak = (red_p - RED_TH).abs().min(-1).values            # closest-to-threshold colour call
    char_weak = torch.where(red_mask, char_margin, torch.ones_like(char_margin)).min(-1).values  # smallest char margin among red

    pred_str = [decode_prediction(char_pred[i], red_mask[i].long()) for i in range(N)]

    # summaries
    print(f"N={N} test images")
    for t in (0.02, 0.05, 0.10, 0.15, 0.20):
        print(f"  colour borderline |P(red)-{RED_TH}|<{t}: {(color_weak < t).sum().item():4d} 图")
    for t in (0.05, 0.10, 0.20, 0.30):
        print(f"  char  borderline primary-margin<{t}: {(char_weak < t).sum().item():4d} 图")

    def montage(order, kind, fname, k=48):
        sel = order[:k].tolist()
        ncol, nrow = 6, (k + 5) // 6
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.0, nrow * 1.35))
        for ax in axes.flat:
            ax.axis("off")
        for j, i in enumerate(sel):
            ax = axes.flat[j]
            ax.imshow(images[i].permute(1, 2, 0).numpy())
            # find the weak position + detail
            if kind == "color":
                pos = int((red_p[i] - RED_TH).abs().argmin())
                detail = f"pos{pos} P(red)={red_p[i, pos]:.2f}"
            else:
                mm = torch.where(red_mask[i], char_margin[i], torch.ones(5))
                pos = int(mm.argmin())
                # top-2 primary chars at that position
                detail = f"pos{pos} m={char_margin[i,pos]:.2f}"
            ax.set_title(f"{ids[i]}  [{pred_str[i] or '∅'}]\n{detail}", fontsize=7)
        fig.suptitle(f"98.98 lowest-confidence TEST ({kind})", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(fname, dpi=110)
        plt.close(fig)
        print("saved", fname)

    color_order = torch.argsort(color_weak)
    char_order = torch.argsort(char_weak)
    montage(color_order, "color", "outputs/lowconf_color.png")
    montage(char_order, "char", "outputs/lowconf_char.png")

    # CSV detail (top 60 of each)
    with open("outputs/lowconf.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "id", "pred", "weak_pos", "P_red_at_pos", "char_margin_at_pos",
                    "glyph_margin_at_pos", "color_weak", "char_weak"])
        for kind, order in (("color", color_order), ("char", char_order)):
            for i in order[:60].tolist():
                if kind == "color":
                    pos = int((red_p[i] - RED_TH).abs().argmin())
                else:
                    mm = torch.where(red_mask[i], char_margin[i], torch.ones(5))
                    pos = int(mm.argmin())
                w.writerow([kind, ids[i], pred_str[i] or "", pos, f"{red_p[i,pos]:.3f}",
                            f"{char_margin[i,pos]:.3f}", f"{glyph_margin[i,pos]:.3f}",
                            f"{color_weak[i]:.3f}", f"{char_weak[i]:.3f}"])
    print("saved outputs/lowconf.csv")


if __name__ == "__main__":
    main()
