"""DINOv3 feature extraction and projection helpers."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from base.config import DINOConfig, ProjectPaths

from .vision_transformer import vit_base


# Buffers the released checkpoint carries that this implementation has no slot
# for. ``qkv.bias_mask`` is all-zero, so dropping it is a no-op. Anything else
# turning up as unexpected means the architecture drifted from the checkpoint
# and the features will be silently wrong — see the LayerScale incident.
_EXPECTED_UNUSED = ("attn.qkv.bias_mask",)


def load_dinov3_vit_base(
    device: torch.device | str,
    weights_file: str | Path | None = None,
    patch_size: int = 16,
    n_storage_tokens: int = 4,
) -> torch.nn.Module:
    """Load the frozen DINOv3 ViT-B/16 witness, verifying the weights actually fit.

    ``n_storage_tokens=4`` matches the released checkpoint. (Measured: running
    with 0 instead changes patch features by cosine 0.998 with no artifact
    tokens, so this is correctness rather than a fix.)
    """
    paths = ProjectPaths()
    weights_file = Path(weights_file) if weights_file is not None else paths.dinov3_weights_file

    if not weights_file.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_file}")

    model = vit_base(patch_size=patch_size, n_storage_tokens=n_storage_tokens)
    checkpoint = torch.load(str(weights_file), map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    result = model.load_state_dict(state_dict, strict=False)

    stray = [k for k in result.unexpected_keys if not k.endswith(_EXPECTED_UNUSED)]
    if result.missing_keys or stray:
        raise RuntimeError(
            f"DINOv3 checkpoint does not match this ViT implementation.\n"
            f"  weights: {weights_file}\n"
            f"  missing (model wants, checkpoint lacks): {result.missing_keys[:8]}\n"
            f"  stray   (checkpoint has, model lacks):   {stray[:8]}\n"
            "Every stray tensor is a trained parameter being thrown away — the "
            "extracted features will not be DINOv3 features."
        )
    return model.to(device).eval()


def compute_patch_aligned_size(image_h: int, image_w: int, max_side: int, patch_size: int) -> tuple[int, int]:
    scale = float(max_side) / float(max(image_h, image_w))
    target_h = max(patch_size, int(round(image_h * scale / patch_size)) * patch_size)
    target_w = max(patch_size, int(round(image_w * scale / patch_size)) * patch_size)
    return target_h, target_w
