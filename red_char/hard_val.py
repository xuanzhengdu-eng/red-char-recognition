"""Test-like hard validation proxy.

The clean held-out val (2500) sits at the noise floor (~7-9 errors), so config
choices tuned on it do not transfer to the platform test set (~50 errors): test
is a *harder distribution* (softer rendering, fainter red, more interference)
than the training images. This script reproduces that gap on TRAIN-DERIVED data
only (no test images, no test labels): it applies out-of-distribution degradation
(blur + resolution loss + faint/uneven red + interference lines + noise) to the
val split, replicated K times per image so the error count is large enough to
compare configs with low relative noise.

Crucially it evaluates at FIXED production thresholds (no per-eval threshold
sweep), so comparing two glyph ensembles here is an honest A/B, not a fit to the
eval set. A config that robustly wins under test-like degradation has a real
reason to transfer to the platform.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as VFv2
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as VF

import config
from augment import _draw_red_lines, _faint_fade, _cutout
from dataset import build_train_dataset, decode_prediction, deterministic_split_indices
from glyph import glyph_probabilities, load_glyph_model
from predict import load_model
from eval_reranker import selective_rerank


def hard_degrade(image: torch.Tensor, level: float) -> torch.Tensor:
    """Apply MILD test-like, partly out-of-distribution degradation.

    Test is only ~2x harder than clean val (~0.5%->~1% err), so degradation must
    be gentle. Colour-preserving: red stays red (faint = lighter red, never a
    hue/channel change). `level` scales both the probability and the strength.
    """
    c, h, w = image.shape
    out = image
    p = level  # at level=1 these are the base probabilities below
    # 1. soft rendering (blur) -- main OOD inflator (training only saw tiny blur)
    if random.random() < 0.35 * p:
        sigma = max(0.1, random.uniform(0.3, 0.8) * level)
        out = VFv2.gaussian_blur(out, kernel_size=5, sigma=sigma)
    # 2. resolution loss (downscale then upscale) -- OOD render/compression softness
    if random.random() < 0.25 * p:
        sf = random.uniform(0.78, 0.92)
        nh, nw = max(8, int(h * sf)), max(8, int(w * sf))
        out = VFv2.resize(out, [nh, nw], antialias=True)
        out = VFv2.resize(out, [h, w], antialias=True)
    # 3. faint / uneven red (model is fairly robust; keep low prob)
    if random.random() < 0.20 * p:
        out = _faint_fade(out)
    # 4. red interference lines
    if random.random() < 0.20 * p:
        out = _draw_red_lines(out, n=random.randint(1, 2))
    # 5. mild occlusion
    if random.random() < 0.08 * p:
        out = _cutout(out, n=1)
    # 6. sensor noise
    out = out + torch.randn_like(out) * (0.008 * level)
    return out.clamp_(0.0, 1.0)


@torch.no_grad()
def run_pipeline(images, primary_models, glyph_models, x_tta, device, color_models=None):
    """char/primary prob from `primary_models`; colour prob from `color_models`
    if given (decoupled), else from `primary_models`."""
    images = images.to(device, non_blocking=True)
    primary = color = None
    shifts = (0, -4, 4) if x_tta else (0,)
    for shift in shifts:
        shifted = images if shift == 0 else VF.affine(
            images, angle=0, translate=[shift, 0], scale=1.0, shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR, fill=[1.0, 1.0, 1.0])
        for model in primary_models:
            char_logits, color_logits = model(shifted)
            cc = F.softmax(char_logits, dim=-1)
            primary = cc if primary is None else primary + cc
            if color_models is None:
                ccol = F.softmax(color_logits, dim=-1)
                color = ccol if color is None else color + ccol
        if color_models is not None:
            for model in color_models:
                _, color_logits = model(shifted)
                ccol = F.softmax(color_logits, dim=-1)
                color = ccol if color is None else color + ccol
    primary = primary / primary.sum(-1, keepdim=True)
    color = color / color.sum(-1, keepdim=True)
    glyph = torch.stack([glyph_probabilities([m], images) for m in glyph_models], 0).mean(0)
    return primary.cpu(), color.cpu(), glyph.cpu()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--glyph-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--color-checkpoints", type=Path, nargs="+", default=None,
                        help="decoupled colour decision: take red/non-red from these models "
                             "(char still from --checkpoints + glyph rerank)")
    parser.add_argument("--reps", type=int, default=4, help="degraded copies per val image")
    parser.add_argument("--level", type=float, default=1.0, help="degradation strength multiplier")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--x-tta", action="store_true")
    parser.add_argument("--primary-margin-max", type=float, default=0.50)
    parser.add_argument("--glyph-margin-min", type=float, default=0.05)
    parser.add_argument("--red-threshold", type=float, default=0.20)
    parser.add_argument("--clean", action="store_true", help="no degradation (sanity reference)")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    device = torch.device(config.DEVICE)
    base = build_train_dataset(cache_in_ram=True)
    _, val_indices = deterministic_split_indices(len(base))
    primary_models = [load_model(p, device) for p in args.checkpoints]
    glyph_models = [load_glyph_model(p, device, use_ema=True) for p in args.glyph_checkpoints]
    color_models = ([load_model(p, device) for p in args.color_checkpoints]
                    if args.color_checkpoints else None)

    # Pre-build the degraded image bank (reproducible), so a later run with a
    # different glyph set sees identical inputs -> honest A/B.
    imgs, char_tgts, color_tgts = [], [], []
    reps = 1 if args.clean else args.reps
    for k in range(reps):
        for idx in val_indices:
            image, char_t, color_t = base[idx]
            if not args.clean:
                random.seed(args.seed + k * 1_000_003 + idx)
                torch.manual_seed(args.seed + k * 1_000_003 + idx)
                image = hard_degrade(image, args.level)
            imgs.append(image)
            char_tgts.append(char_t)
            color_tgts.append(color_t)
    char_target = torch.stack(char_tgts)
    color_target = torch.stack(color_tgts)
    red_mask = color_target.bool()
    true_strings = [decode_prediction(char_target[i], color_target[i]) for i in range(len(char_target))]

    primary_probs, color_probs, glyph_probs = [], [], []
    for i in range(0, len(imgs), args.batch_size):
        batch = torch.stack(imgs[i:i + args.batch_size])
        p, c, g = run_pipeline(batch, primary_models, glyph_models, args.x_tta, device,
                               color_models=color_models)
        primary_probs.append(p); color_probs.append(c); glyph_probs.append(g)
    primary_prob = torch.cat(primary_probs)
    color_prob = torch.cat(color_probs)
    glyph_prob = torch.cat(glyph_probs)

    n = len(char_target)
    # primary-only (no rerank) reference
    char_primary = primary_prob.argmax(-1)
    # selective rerank at FIXED production thresholds
    char_rerank = selective_rerank(primary_prob, glyph_prob, top_k=3,
                                   primary_margin_max=args.primary_margin_max,
                                   glyph_margin_min=args.glyph_margin_min)
    color_pred = color_prob[..., config.RED_INDEX].ge(args.red_threshold).long()

    def exact(char_pred):
        return sum(decode_prediction(char_pred[i], color_pred[i]) == true_strings[i] for i in range(n))

    red_acc_primary = char_primary.eq(char_target)[red_mask].float().mean().item()
    red_acc_rerank = char_rerank.eq(char_target)[red_mask].float().mean().item()
    ex_primary = exact(char_primary)
    ex_rerank = exact(char_rerank)
    mode = "CLEAN" if args.clean else f"HARD lvl={args.level} reps={reps}"
    glyph_n = len(args.glyph_checkpoints)
    print(f"[{args.tag}] {mode} n={n} glyph_models={glyph_n} "
          f"| primary-only exact={ex_primary}/{n}={ex_primary/n:.5f} red_char_acc={red_acc_primary:.5f} "
          f"| selective-rerank exact={ex_rerank}/{n}={ex_rerank/n:.5f} ({n-ex_rerank} err) "
          f"red_char_acc={red_acc_rerank:.5f}")

    # --- error decomposition (uses the selective-rerank char prediction) ---
    # For each wrong sample, attribute the error to COLOR (red/non-red) and/or CHAR
    # (glyph identity) by oracle-substituting one component at a time.
    pred_str  = [decode_prediction(char_rerank[i], color_pred[i])   for i in range(n)]
    fix_color = [decode_prediction(char_rerank[i], color_target[i]) for i in range(n)]  # perfect color
    fix_char  = [decode_prediction(char_target[i], color_pred[i])   for i in range(n)]  # perfect char
    n_wrong = color_only = char_only = both = 0
    for i in range(n):
        if pred_str[i] == true_strings[i]:
            continue
        n_wrong += 1
        ok_if_color = (fix_color[i] == true_strings[i])  # fixing color alone solves it
        ok_if_char  = (fix_char[i]  == true_strings[i])  # fixing char alone solves it
        if ok_if_color and not ok_if_char:
            color_only += 1
        elif ok_if_char and not ok_if_color:
            char_only += 1
        else:
            both += 1  # both wrong, or interaction
    print(f"[{args.tag}] error-decomp: wrong={n_wrong} | color-only={color_only} "
          f"({color_only/max(1,n_wrong):.0%}) char-only={char_only} ({char_only/max(1,n_wrong):.0%}) "
          f"both/other={both} ({both/max(1,n_wrong):.0%})")


if __name__ == "__main__":
    main()
