from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

import config
from augment import TrainAugment
from dataset import RedCharDataset
from model import ResidualSEStage


GLYPH_CROP_WIDTH = 64
PAIR_GROUPS = ("1ILTVP", "2Z7T", "3S568E", "CGOQ", "7NY", "PF")


def extract_glyph_crops(images: torch.Tensor, crop_width: int = GLYPH_CROP_WIDTH,
                        offset: int = 0) -> torch.Tensor:
    """Extract overlapping crops centred on the five nominal character slots.

    Padding keeps edge slots the same size as middle slots. The crop is wider
    than one 40-pixel slot so position jitter does not clip the target glyph.
    ``offset`` shifts the crop window horizontally (for crop-offset TTA, to be
    robust to characters that sit off their nominal slot centre).
    """
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4 or images.shape[-2:] != (config.IMAGE_HEIGHT, config.IMAGE_WIDTH):
        raise ValueError(f"expected [B,3,60,200], got {tuple(images.shape)}")
    half = crop_width // 2
    padded = F.pad(images, (half, half, 0, 0), value=1.0)
    crops = []
    for position in range(config.NUM_POSITIONS):
        center = (position * 2 + 1) * config.IMAGE_WIDTH // (2 * config.NUM_POSITIONS) + offset
        center = max(0, min(center, padded.shape[-1] - crop_width))
        crops.append(padded[..., center : center + crop_width])
    return torch.stack(crops, dim=1)


class GlyphDataset(Dataset):
    """Position-level view over full images, optionally restricted to red glyphs."""

    def __init__(
        self,
        base: RedCharDataset,
        image_indices: list[int],
        red_only: bool = True,
        augment: bool = False,
        red_line_p: float = 0.0,
        cutout_p: float = 0.0,
        faint_p: float = 0.0,
        occlude_p: float = 0.0,
        crop_width: int = GLYPH_CROP_WIDTH,
        boost_chars: str = "",
        boost_factor: int = 1,
    ) -> None:
        self.base = base
        self.crop_width = crop_width
        self.items: list[tuple[int, int]] = []
        boost_set = set(boost_chars)
        for image_idx in image_indices:
            sample = base.samples[image_idx]
            if sample.color is None:
                raise ValueError("glyph dataset requires labelled samples")
            for position, color in enumerate(sample.color):
                if not red_only or color == "r":
                    # oversample hard confusion-group chars to focus capacity
                    reps = boost_factor if sample.all_label[position] in boost_set else 1
                    for _ in range(reps):
                        self.items.append((image_idx, position))
        self.transform = (
            TrainAugment(translate=0.08, scale=(0.94, 1.06), degrees=5.0, noise_std=0.015,
                         red_line_p=red_line_p, cutout_p=cutout_p, faint_p=faint_p,
                         occlude_p=occlude_p)
            if augment
            else None
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_idx, position = self.items[index]
        image, char_target, _ = self.base[image_idx]
        crop = extract_glyph_crops(image, crop_width=self.crop_width)[0, position]
        if self.transform is not None:
            crop = self.transform(crop)
        return crop, char_target[position]


# --- DenseNet backbone (a genuinely DIFFERENT architecture from the residual+SE
# backbone, for decorrelated ensemble diversity: dense connectivity = each layer
# sees all previous feature maps) ---------------------------------------------
class _DenseLayer(nn.Module):
    def __init__(self, in_ch: int, growth: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 4 * growth, 1, bias=False),
            nn.BatchNorm2d(4 * growth), nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth, growth, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return torch.cat([x, self.net(x)], dim=1)


class _Transition(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
                                 nn.Conv2d(in_ch, out_ch, 1, bias=False))
        self.pool = nn.AvgPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.net(x))


class DenseGlyphBackbone(nn.Module):
    """Compact DenseNet for a 60x64 glyph crop; ends at 30x32 (matches the SE
    backbone's hi-res sweet spot) so only the connectivity differs."""

    def __init__(self, base_in: int, growth: int = 16) -> None:
        super().__init__()
        c = 2 * growth
        layers = [nn.Conv2d(base_in, c, 3, padding=1, bias=False)]  # stem, keeps 60x64
        block_cfg = [(6, True), (8, False), (8, False)]  # (n_layers, pool_after); 1 pool -> 30x32
        for n, pool in block_cfg:
            for _ in range(n):
                layers.append(_DenseLayer(c, growth)); c += growth
            layers.append(_Transition(c, c // 2, pool=pool)); c //= 2
        layers += [nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x):
        return self.net(x)


_RESNET18_CACHE = Path("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth").expanduser()


def _build_resnet18_backbone() -> nn.Module:
    """ResNet-18 conv trunk (through layer4 -> [B,512,h,w]), initialised from the
    cached ImageNet weights when present (offline-safe; falls back to random)."""
    import torchvision
    net = torchvision.models.resnet18(weights=None)
    if _RESNET18_CACHE.exists():
        state = torch.load(_RESNET18_CACHE, map_location="cpu")
        net.load_state_dict(state)
    return nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                         net.layer1, net.layer2, net.layer3, net.layer4)


class GlyphNet(nn.Module):
    """Shared high-resolution classifier for one approximately centred glyph.

    ``hires=True`` keeps one more pooling stage's worth of resolution (30x32
    feature map instead of 15x16) so thin-stroke confusions (I/1/L/T/J, E/F)
    keep their distinguishing serif/length detail. Feature dim is inferred so
    crop width / pooling can change freely. ``backbone='dense'`` swaps the
    residual+SE backbone for a DenseNet (architectural diversity for ensembling).
    """

    def __init__(self, dropout: float = 0.2, input_mode: str = "rgb",
                 hires: bool = False, crop_width: int = GLYPH_CROP_WIDTH,
                 head_mode: str = "flat", n_pool: int | None = None,
                 backbone: str = "se") -> None:
        super().__init__()
        if input_mode not in {"rgb", "red", "red2", "binred"}:
            raise ValueError(f"unknown glyph input mode: {input_mode}")
        if head_mode not in {"flat", "gap"}:
            raise ValueError(f"unknown head_mode: {head_mode}")
        if backbone not in {"se", "dense", "resnet18"}:
            raise ValueError(f"unknown backbone: {backbone}")
        self.input_mode = input_mode
        self.hires = hires
        self.crop_width = crop_width
        self.head_mode = head_mode
        self.backbone_type = backbone
        widths = (48, 96, 192, 256)
        # n_pool overrides hires: number of leading stages that downsample.
        # n_pool=0 keeps full 60x64 resolution (max detail; needs GAP head).
        if n_pool is None:
            n_pool = 1 if hires else 2
        self.n_pool = n_pool
        base_in = 5 if input_mode in {"red", "red2", "binred"} else 3
        if backbone == "resnet18":
            # ImageNet-pretrained ResNet-18 as a genuinely heterogeneous member.
            # Natural-image features decorrelate from our from-scratch nets; the
            # crop is fed as RGB (resized + ImageNet-normalised in features()).
            if head_mode != "gap":
                raise ValueError("resnet18 backbone requires head_mode=gap")
            self.backbone = _build_resnet18_backbone()
            head_in = 512
            self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        elif backbone == "dense":
            if head_mode != "gap":
                raise ValueError("dense backbone requires head_mode=gap")
            self.backbone = DenseGlyphBackbone(base_in)
            head_in = self.backbone.out_channels
        else:
            pools = tuple(i < n_pool for i in range(len(widths)))
            stages = []
            in_channels = base_in
            for out_channels, pool in zip(widths, pools):
                stages.append(ResidualSEStage(in_channels, out_channels, pool=pool))
                in_channels = out_channels
            self.backbone = nn.Sequential(*stages)
            head_in = widths[-1]
        if head_mode == "gap":
            # Global average pooling keeps the hi-res conv detail but a tiny head
            # (no 30k-wide FC), so deeper/finer features don't overfit.
            self.reduce = nn.Identity()
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(head_in, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, config.NUM_CHARS),
            )
        else:
            self.reduce = nn.Sequential(
                nn.Conv2d(widths[-1], 32, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, base_in, config.IMAGE_HEIGHT, crop_width)
                feat_dim = self.reduce(self.backbone(dummy)).flatten(1).shape[1]
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_dim, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, config.NUM_CHARS),
            )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_type == "resnet18":
            # crop is [B,3,H,W] in [0,1]; resize to ImageNet-ish size + normalise
            x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
            x = (x - self.imagenet_mean) / self.imagenet_std
            return self.head[:-1](self.reduce(self.backbone(x)))
        if self.input_mode in {"red", "red2", "binred"}:
            red = x[:, 0:1]
            redness = (red - x[:, 1:3].amax(dim=1, keepdim=True)).relu()
            if self.input_mode == "binred":
                # hard color-binarisation (the ttocr article's idea): red pixel -> 1
                redness = (redness > 0.10).float()
            if self.input_mode == "red2":
                # intensity-robust: normalise redness per-crop by its own max so
                # FAINT red strokes (e.g. the light right stroke of a V) stay
                # visible relative to the strongest red, instead of being
                # suppressed by the absolute-difference redness (fixes V->I).
                m = redness.amax(dim=(2, 3), keepdim=True)
                redness = redness / (m + 1e-4)
            darkness = 1.0 - x.mean(dim=1, keepdim=True)
            x = torch.cat([x, redness, darkness], dim=1)
        return self.head[:-1](self.reduce(self.backbone(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head[-1](self.features(x))


class PairGlyphNet(nn.Module):
    """Glyph network with dedicated heads for known hard confusion groups."""

    def __init__(self, input_mode: str = "rgb") -> None:
        super().__init__()
        self.base = GlyphNet(input_mode=input_mode)
        self.pair_heads = nn.ModuleList(
            [nn.Linear(512, len(chars)) for chars in PAIR_GROUPS]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = self.base.features(x)
        full_logits = self.base.head[-1](features)
        return full_logits, [head(features) for head in self.pair_heads]


def load_glyph_model(checkpoint: Path, device: torch.device, use_ema: bool = True) -> GlyphNet:
    payload = torch.load(checkpoint, map_location=device)
    model = GlyphNet(input_mode=payload.get("input_mode", "rgb"),
                     hires=payload.get("hires", False),
                     head_mode=payload.get("head_mode", "flat"),
                     crop_width=payload.get("crop_width", GLYPH_CROP_WIDTH),
                     n_pool=payload.get("n_pool", None),
                     backbone=payload.get("backbone", "se")).to(device)
    state = payload.get("ema_state_dict") if use_ema else None
    model.load_state_dict(state if state is not None else payload["state_dict"])
    model.eval()
    return model


def load_pair_glyph_model(
    checkpoint: Path, device: torch.device, use_ema: bool = True
) -> PairGlyphNet:
    payload = torch.load(checkpoint, map_location=device)
    model = PairGlyphNet(input_mode=payload.get("input_mode", "rgb")).to(device)
    state = payload.get("ema_state_dict") if use_ema else None
    model.load_state_dict(state if state is not None else payload["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def glyph_probabilities(models: list[GlyphNet], images: torch.Tensor,
                        tta_offsets: tuple[int, ...] = (0,)) -> torch.Tensor:
    """Return averaged probabilities shaped [B, 5, 36].

    ``tta_offsets`` averages over several horizontal crop offsets (crop-offset
    TTA): robustness to characters that sit off their nominal slot centre.
    """
    batch_size = images.shape[0]
    crop_width = getattr(models[0], "crop_width", GLYPH_CROP_WIDTH)
    probabilities = None
    n = 0
    for off in tta_offsets:
        crops = extract_glyph_crops(images, crop_width=crop_width, offset=off).flatten(0, 1)
        for model in models:
            current = model(crops).softmax(dim=-1)
            probabilities = current if probabilities is None else probabilities + current
            n += 1
    if probabilities is None:
        raise ValueError("at least one glyph model is required")
    return (probabilities / n).view(batch_size, config.NUM_POSITIONS, config.NUM_CHARS)
