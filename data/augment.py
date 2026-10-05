"""Photometric augmentation, ported from MVSFormer++'s ``datasets/color_jittor.py``.

Two rules make this MVS-specific rather than a generic torchvision pipeline:

1. **One parameter draw per sample, shared by every view.** Plane-sweep matching
   assumes photo-consistency across views; jittering each view independently
   would break exactly the signal the cost volume measures. MVSFormer++ draws
   ``fn_idx`` and the four factors once before its view loop — same here.
2. **Images stay in 0-255.** ``UprMVSNet.forward`` divides by 255 itself and the
   FPN has no input normalisation, so this returns the same range it was given
   (MVSFormer++ normalises inside the dataset instead).

The depth prior is cached from un-jittered images and is deliberately NOT
re-jittered: a prior that stays put while the appearance moves is the realistic
case (the prior model saw one particular exposure), and it gives SPRE a reason
to distrust appearance-only evidence. It does mean the DINOv3 witness sees a
slightly different image than the prior was built from.
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.functional as TF


class PhotometricAug:
    """Brightness / contrast / saturation / hue jitter + random gamma.

    Defaults are MVSFormer++'s DTU settings (``config/mvsformer++.json``:
    ``aug_args``). ``draw()`` returns one parameter set to apply to all views.
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.1,
        saturation: float = 0.1,
        hue: float = 0.05,
        min_gamma: float = 0.9,
        max_gamma: float = 1.1,
    ) -> None:
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        self.min_gamma = float(min_gamma)
        self.max_gamma = float(max_gamma)

    def draw(self, rng: np.random.Generator) -> dict:
        return {
            # random op order, as in MVSFormer++ (torch.randperm(4))
            "order": rng.permutation(4).tolist(),
            "brightness": float(rng.uniform(max(0.0, 1 - self.brightness), 1 + self.brightness)),
            "contrast": float(rng.uniform(max(0.0, 1 - self.contrast), 1 + self.contrast)),
            "saturation": float(rng.uniform(max(0.0, 1 - self.saturation), 1 + self.saturation)),
            "hue": float(rng.uniform(-self.hue, self.hue)),
            "gamma": float(rng.uniform(self.min_gamma, self.max_gamma)),
        }

    @staticmethod
    def apply(img: np.ndarray, params: dict) -> np.ndarray:
        """``img``: HWC, 0-255 (uint8 or float). Returns HWC float32, 0-255."""
        # asarray(PIL) is read-only; converting to float32 here gives torch a
        # writable buffer (and folds in the .float() we would do anyway).
        t = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
        for op in params["order"]:
            if op == 0:
                t = TF.adjust_brightness(t, params["brightness"])
            elif op == 1:
                t = TF.adjust_contrast(t, params["contrast"])
            elif op == 2:
                t = TF.adjust_saturation(t, params["saturation"])
            else:
                t = TF.adjust_hue(t, params["hue"])
        # gamma after the jitter, on [0, 1], clipped — MVSFormer++'s RandomGamma
        t = t.clamp(0.0, 1.0).pow(params["gamma"]).clamp(0.0, 1.0)
        return (t * 255.0).permute(1, 2, 0).numpy().astype(np.float32)


# MVSFormer++'s full DTU scale list (config/mvsformer++.json). Kept here as the
# reference set; base.config picks a subset that fits this network's memory —
# stage 3 runs a cost volume at stride 1, so pixels cost far more here than in
# their 4-stage/stride-8 design.
MVSFORMERPP_SCALES: tuple[tuple[int, int], ...] = (
    (512, 640), (512, 704), (512, 768), (576, 704), (576, 768),
    (576, 832), (640, 832), (640, 896), (640, 960), (704, 896),
    (704, 960), (704, 1024), (768, 960), (768, 1024), (768, 1088),
    (832, 1024), (832, 1088), (832, 1152), (896, 1152), (896, 1216),
    (896, 1280), (960, 1216), (960, 1280), (960, 1344), (1024, 1280),
)


def resize_scale_for_crop(
    crop_h: int,
    crop_w: int,
    src_h: int,
    src_w: int,
    enlarge: float,
    lo: float = 0.45,
    hi: float = 1.0,
) -> float:
    """MVSFormer++'s scale rule: resize so the frame is ``enlarge`` x the crop.

    The margin above 1.0 is what gives the random crop somewhere to move; the
    [lo, hi] clamp stops tiny crops from downsampling the source into mush or
    large ones from upsampling past the native resolution.

    The ``+ 0.5`` floor is not in their code and is needed here: ``pre_resize``
    sizes with ``int(src * scale)``, so a scale that lands on e.g. 959.9999999
    truncates to one pixel below the crop, the crop slice comes back short, and
    the batch fails to stack. Requiring ``src * scale >= crop + 0.5`` makes the
    truncated size at least ``crop`` for any float error of that size.
    """
    s = float(np.clip(max((crop_h * enlarge) / src_h, (crop_w * enlarge) / src_w), lo, hi))
    return max(s, (crop_h + 0.5) / src_h, (crop_w + 0.5) / src_w)
