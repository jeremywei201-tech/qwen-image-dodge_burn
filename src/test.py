# -*- coding: utf-8 -*-
"""
Qwen-Image-Dodge&Burn — Inference entry point
=============================================

Thin runner around :class:`DodgeBurnPipeline` (defined in
``inference_pipeline.py``). Loads the model once and retouches one image.

The basic image is first converted to a black & white, S-curve-contrasted
image (``contrast_strength`` default 10.0) which is fed to the model so it
identifies shadows/highlights better; the original colour image is what the
predicted layers and face mask are composited onto (see DEV.md "Updates").

Inference parameters
---------------------
Required:
    image_path : str    — path of the input image to retouch.

Optional (defaults shown):
    need_high_freq : bool  = False  — whether to composite the high-frequency
                                      layer. Compositing it is usually not very
                                      pleasing, so it is off by default.
    high_coeff     : float = 1.0    — strength of the high-frequency layer.
    low_coeff      : float = 1.0    — strength of the low-frequency (DNB) layer.

Usage (CLI)
-----------
    python3 infer.py --image_path /path/to/photo.png
    python3 infer.py --image_path /path/to/photo.png \
        --need_high_freq --high_coeff 0.6 --low_coeff 1.2

Usage (programmatic)
--------------------
    from infer import DodgeBurnInfer
    runner = DodgeBurnInfer()                  # loads weights once
    out = runner.infer({"image_path": "photo.png", "low_coeff": 1.2})
"""

import argparse
from pathlib import Path

import torch

from inference_pipeline import DodgeBurnPipeline


# ---------------------------------------------------------------------
# Default weight locations (override via constructor / CLI).
# ---------------------------------------------------------------------
DEFAULT_PRETRAINED_PATH = "/data/qwen_weights"
DEFAULT_FACE_PARSING_WEIGHTS = "/data/qwen-image-dodge_burn/src/model/79999_iter.pth"
DEFAULT_OUTPUT_DIR = "/data/qwen-image-dodge_burn/qwen_dnb_output"
DEFAULT_CONTRAST_STRENGTH = 10.0  # S-curve strength for the B&W model input
DEFAULT_PROMPT = (
    "Add a neutral gray layer to the character in the picture to make the "
    "skin light and shadow of the character smooth and textured. At the same "
    "time, keep the facial contours of the character unchanged"
)

# Inference-parameter defaults (the spec requested in the task).
DEFAULT_INFER_PARAMS = {
    "need_high_freq": False,
    "high_coeff": 1.0,
    "low_coeff": 1.0,
}


class DodgeBurnInfer:
    """Loads the Dodge&Burn pipeline once, then retouches images on demand."""

    def __init__(self,
                 pretrained_path=DEFAULT_PRETRAINED_PATH,
                 face_parsing_weights=DEFAULT_FACE_PARSING_WEIGHTS,
                 output_dir=DEFAULT_OUTPUT_DIR,
                 prompt=DEFAULT_PROMPT,
                 contrast_strength=DEFAULT_CONTRAST_STRENGTH,
                 device=None,
                 load_qwen=True):
        self.output_dir = output_dir
        self.prompt = prompt
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DodgeBurnInfer] device: {self.device}")

        self.pipe = DodgeBurnPipeline.from_pretrained(
            pretrained_path=pretrained_path,
            face_parsing_weights=face_parsing_weights,
            device=self.device,
            load_qwen=load_qwen,
            contrast_strength=contrast_strength,
        )

    def infer(self, params):
        """Retouch a single image.

        Args:
            params: dict with the required key ``image_path`` and optional
                keys ``need_high_freq`` / ``high_coeff`` / ``low_coeff``.
        Returns:
            Path to the saved retouched image.
        """
        if "image_path" not in params:
            raise ValueError("`image_path` is required.")
        image_path = params["image_path"]
        if not Path(image_path).exists():
            raise FileNotFoundError(f"image_path not found: {image_path}")

        # Merge caller params over the defaults.
        cfg = dict(DEFAULT_INFER_PARAMS)
        for k in DEFAULT_INFER_PARAMS:
            if k in params and params[k] is not None:
                cfg[k] = params[k]

        out_path = self.pipe.run_on_image(
            basic_path=image_path,
            prompt=self.prompt,
            out_dir=self.output_dir,
            need_high_freq=bool(cfg["need_high_freq"]),
            high_coeff=float(cfg["high_coeff"]),
            low_coeff=float(cfg["low_coeff"]),
        )
        print(f"[DodgeBurnInfer] {image_path} -> {out_path} "
              f"(need_high_freq={cfg['need_high_freq']}, "
              f"high_coeff={cfg['high_coeff']}, low_coeff={cfg['low_coeff']})")
        return out_path


def parse_args():
    p = argparse.ArgumentParser(
        description="Qwen-Image-Dodge&Burn single-image inference.")
    # Required inference parameter.
    p.add_argument("--image_path", type=str, required=True,
                   help="Path of the input image to retouch.")
    # Default inference parameters.
    p.add_argument("--need_high_freq", action="store_true",
                   help="Composite the high-frequency layer (default off; "
                        "compositing it is usually not very pleasing).")
    p.add_argument("--high_coeff", type=float, default=DEFAULT_INFER_PARAMS["high_coeff"],
                   help="High-frequency layer strength (default 1.0).")
    p.add_argument("--low_coeff", type=float, default=DEFAULT_INFER_PARAMS["low_coeff"],
                   help="Low-frequency (DNB) layer strength (default 1.0).")
    # Weight / output locations.
    p.add_argument("--pretrained_model_name_or_path", type=str,
                   default=DEFAULT_PRETRAINED_PATH)
    p.add_argument("--face_parsing_weights", type=str,
                   default=DEFAULT_FACE_PARSING_WEIGHTS)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--contrast_strength", type=float,
                   default=DEFAULT_CONTRAST_STRENGTH,
                   help="S-curve contrast strength for the B&W model input "
                        "(default 10.0).")
    return p.parse_args()


def main():
    args = parse_args()

    runner = DodgeBurnInfer(
        pretrained_path=args.pretrained_model_name_or_path,
        face_parsing_weights=args.face_parsing_weights,
        output_dir=args.output_dir,
        contrast_strength=args.contrast_strength,
    )

    runner.infer({
        "image_path": args.image_path,
        "need_high_freq": args.need_high_freq,
        "high_coeff": args.high_coeff,
        "low_coeff": args.low_coeff,
    })


if __name__ == "__main__":
    main()
