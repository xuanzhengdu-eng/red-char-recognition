"""Adaptive character localization (direction 1).

Fixed 5-slot cropping (centers 20/60/100/140/180, width 64) grabs neighbours and
mis-aligns drifted characters (confirmed: 01956 M/L bled into the I crop). This
refines each slot centre to where the character actually is, using a line-robust
column ink-profile: a character column has MANY ink pixels (tall stroke), a thin
interference line contributes only 1-3 px per column, so summing ink over y
naturally emphasises characters over lines.
"""
from __future__ import annotations

import numpy as np
import torch

import config

NOMINAL = [(2 * k + 1) * config.IMAGE_WIDTH // (2 * config.NUM_POSITIONS) for k in range(5)]  # 20..180


def detect_centers(img_3hw: torch.Tensor, search=16, bias_sigma=12.0) -> list[int]:
    """img: [3,H,W] in [0,1] -> 5 refined x-centres. Centre-biased centroid of the
    column ink-profile within each nominal slot (clamped near the nominal centre)."""
    a = img_3hw.permute(1, 2, 0).numpy() if isinstance(img_3hw, torch.Tensor) else img_3hw
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    sat = a.max(-1) - a.min(-1)
    dark = 1.0 - a.mean(-1)
    ink = ((sat > 0.20) | (dark > 0.45)).astype(np.float32)
    col = ink[3:57, :].sum(0)  # [W] ink count per column (tall char cols >> thin line cols)
    centers = []
    xs = np.arange(config.IMAGE_WIDTH)
    for c0 in NOMINAL:
        lo, hi = max(0, c0 - search), min(config.IMAGE_WIDTH, c0 + search)
        w = col[lo:hi] * np.exp(-0.5 * ((xs[lo:hi] - c0) / bias_sigma) ** 2)  # centre-biased
        if w.sum() < 3:
            centers.append(c0)
        else:
            cx = int(round((xs[lo:hi] * w).sum() / w.sum()))
            centers.append(max(c0 - search, min(c0 + search, cx)))
    return centers


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from PIL import Image
    ex = ["01956.png", "01043.png", "00877.png", "02363.png", "01812.png", "03432.png"]
    fig, ax = plt.subplots(len(ex), 1, figsize=(8, len(ex) * 1.5))
    for a_, t in zip(ax, ex):
        im = np.asarray(Image.open(config.TEST_IMAGES / t).convert("RGB"), dtype=np.float32) / 255.
        cen = detect_centers(torch.from_numpy(im).permute(2, 0, 1))
        a_.imshow(im); a_.axis("off"); a_.set_title(f"{t}  fixed=cyan  detected=red", fontsize=8, loc="left")
        for c in NOMINAL: a_.axvline(c, color="cyan", lw=0.6)
        for c in cen: a_.axvline(c, color="red", lw=1.2, ls="--")
    fig.tight_layout(); fig.savefig("outputs/adaptive_centers.png", dpi=130)
    print("saved outputs/adaptive_centers.png")
    print("detected centres:")
    for t in ex:
        im = np.asarray(Image.open(config.TEST_IMAGES / t).convert("RGB"), dtype=np.float32) / 255.
        print(f"  {t}: {detect_centers(torch.from_numpy(im).permute(2,0,1))}  (nominal {NOMINAL})")
