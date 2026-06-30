# -*- coding: utf-8 -*-
"""
Qwen-Image-Dodge&Burn — Inference Pipeline
===========================================

End-to-end inference wrapper for the Qwen-Image-Dodge&Burn model
(a fine-tune of Qwen-Image-Layered specialised for photographic
dodge & burn retouching).

Workflow (see ../../DEV.md):
    1. Run the Qwen layered pipeline on a basic image to infer two layers:
         - layer_0 : high-frequency  texture residual  (neutral-gray, 0.5-centred)
         - layer_1 : low-frequency   light/shadow (DNB) (neutral-gray, 0.5-centred)
    2. Extract a face mask with the BiSeNet-based `FaceParsing` class so the
       retouching stays strictly inside the face.
    3. Linearly superimpose both layers onto the basic image, gated by the
       face mask, producing the final retouched result.

The layer encoding and the linear-light superposition formula mirror the
training-time decomposition in ``src/decompose/dnb_decompose.py`` so that
inference is the exact inverse of decomposition:

    decompose:  layer = 0.5 + (retouch - raw) / scale          (clipped)
    compose:    result = raw + (layer - 0.5)                    (linear light)

Both layers are additive 0.5-centred offsets, so the two-layer composite is:

    result = raw + [(high - 0.5) + (low - 0.5)] * face_mask

This device may not have a GPU; the script is written so the heavy diffusion
step can be skipped (``--skip_infer``) and run against pre-computed layer PNGs
for debugging the compositing stage independently.
"""

import os
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms

# BiSeNet face-parsing network (same one used at decomposition time).
# The model package lives in ../model; we add it to sys.path inside
# FaceParsing so this script stays runnable from any working directory.
import sys

_MODEL_DIR = Path(__file__).resolve().parent / "model"


# =====================================================================
# Black & white + S-curve contrast preprocessing
# =====================================================================
def create_s_curve_lut(strength=10.0):
    """Build a 256-entry S-curve LUT for contrast scaling.

    Ported from ``src/decompose/convert_to_black_white_with_scaling.py``.
    A sigmoid centred at mid-gray steepens shadows/highlights so the model
    can identify light & shadow more easily. ``strength == 0`` is a no-op
    (linear) mapping.
    """
    if strength == 0 or abs(strength) < 1e-5:
        return np.arange(256, dtype=np.uint8)

    x = np.arange(256)
    x_normalized = (x / 255.0) - 0.5
    y_normalized = 1 / (1 + np.exp(-strength * x_normalized))

    y_min = y_normalized.min()
    y_max = y_normalized.max()
    lut = 255.0 * (y_normalized - y_min) / (y_max - y_min)
    return np.clip(lut, 0, 255).astype(np.uint8)


def convert_to_black_white_with_scaling(image_bgr, contrast_strength=10.0):
    """Convert a BGR image to grayscale and apply the S-curve contrast LUT.

    Returns a single-channel uint8 image — the model input for Dodge&Burn.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lut = create_s_curve_lut(strength=contrast_strength)
    return cv2.LUT(gray, lut)


# =====================================================================
# Face parsing (BiSeNet) — wrapped exactly as in dnb_decompose.py
# =====================================================================
class FaceParsing:
    """BiSeNet-based face parser.

    Produces a 19-class face-parsing map from a BGR image.  Only used at
    inference time to derive the face mask that gates the dodge & burn layers.
    """

    # BiSeNet/CelebAMask-HQ label ids that make up the facial skin region we
    # want to retouch. Matches `core_face_parts` in dnb_decompose.py.
    CORE_FACE_PARTS = [1, 2, 3, 4, 5, 10, 11, 12, 13]

    def __init__(self, model_path, device=None):
        if str(_MODEL_DIR) not in sys.path:
            sys.path.insert(0, str(_MODEL_DIR))
        from model import BiSeNet  # noqa: E402  (BiSeNet defined in ../model/model.py)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.net = BiSeNet(n_classes=19)
        state = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(state)
        self.net.to(self.device)
        self.net.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def infer(self, img_bgr):
        """Return a (H, W) uint8 parsing map at the input resolution."""
        h, w = img_bgr.shape[:2]

        scale = 512 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = cv2.resize(img_bgr, (new_w, new_h))
        canvas = np.zeros((512, 512, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = img_resized

        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.net(tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

        parsing = parsing[:new_h, :new_w]
        parsing = cv2.resize(parsing.astype(np.uint8), (w, h),
                             interpolation=cv2.INTER_NEAREST)
        return parsing


# =====================================================================
# Dodge & Burn inference pipeline
# =====================================================================
class DodgeBurnPipeline:
    """Wraps Qwen layer inference + face parsing + linear-light compositing.

    Layer convention (chosen to match the decompose stage):
        layer_0 -> high-frequency texture residual  (0.5 == no change)
        layer_1 -> low-frequency  light/shadow (DNB) (0.5 == no change)

    Both layers are neutral-gray (0.5-centred) additive offsets, so the
    final result is the linear superposition

        result = raw + [(high - 0.5) + (low - 0.5)] * face_mask

    Model input vs. compositing base (see DEV.md "Updates"):
        - The Qwen model is fed a *black & white, S-curve-contrasted* version
          of the basic image so it can identify shadows / highlights better.
        - The *original* (colour) basic image is what the predicted layers and
          face mask are composited onto.
    """

    HIGH_FREQ_IDX = 0
    LOW_FREQ_IDX = 1

    def __init__(self, qwen_pipeline, face_parser, mp_face_mesh=None,
                 contrast_strength=10.0):
        self.pipeline = qwen_pipeline      # diffusers QwenImageLayeredPipeline (or None)
        self.face_parser = face_parser     # FaceParsing instance
        self.mp_face_mesh = mp_face_mesh   # optional mediapipe FaceMesh for refinement
        self.contrast_strength = contrast_strength  # S-curve strength for B&W input

    # -----------------------------------------------------------------
    # Convenience factory: build a ready-to-use pipeline from weight paths
    # -----------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, pretrained_path,
                        face_parsing_weights, device=None,
                        load_qwen=True, use_face_mesh=True,
                        contrast_strength=10.0,
                        torch_dtype=torch.bfloat16):
        """Build a DodgeBurnPipeline from on-disk weights.

        Args:
            pretrained_path: Qwen-Image-DODGE-BURN base weights dir.
            face_parsing_weights: BiSeNet checkpoint for FaceParsing.
            device: torch device; auto-detected (cuda/cpu) when None.
            load_qwen: load the diffusion pipeline (set False to composite
                pre-computed layers only — e.g. GPU-less debugging).
            use_face_mesh: enable MediaPipe FaceMesh refinement when available.
            contrast_strength: S-curve strength for the B&W model input
                (DEV.md default 10.0).
        """
        device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        qwen_pipeline = None
        if load_qwen:
            qwen_pipeline = load_qwen_pipeline(
                pretrained_path, device, torch_dtype=torch_dtype)

        face_parser = FaceParsing(face_parsing_weights, device=str(device))
        face_mesh = build_face_mesh() if use_face_mesh else None
        return cls(qwen_pipeline, face_parser, face_mesh,
                   contrast_strength=contrast_strength)

    # -----------------------------------------------------------------
    # Step 1 — infer the two dodge & burn layers with the Qwen model
    # -----------------------------------------------------------------
    def infer_layers(self, image_rgba, prompt, negative_prompt=" ",
                     num_inference_steps=50, resolution=1024, seed=777,
                     true_cfg_scale=1.0, layers=2):
        """Run the Qwen layered pipeline; return a list of PIL layer images.

        ``image_rgba`` is a PIL.Image in RGBA mode. Returns the raw per-layer
        PIL images exactly as produced by the diffusion model.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "Qwen pipeline not loaded. Either load it (GPU required) or run "
                "with --skip_infer against pre-computed layer PNGs."
            )

        device = self.pipeline.device
        inputs = {
            "image": image_rgba,
            "generator": torch.Generator(device=device).manual_seed(seed),
            "prompt": prompt,
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "num_images_per_prompt": 1,
            "layers": layers,
            "resolution": resolution,
            "cfg_normalize": True,
            "use_en_prompt": False,
            "random_mode": 0,
        }

        with torch.inference_mode():
            autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"
            with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
                output = self.pipeline(**inputs)
        # output.images[0] is the list of per-layer PIL images for this prompt.
        return list(output.images[0])

    # -----------------------------------------------------------------
    # Step 2 — derive the face mask (inference-only port of dnb_decompose)
    # -----------------------------------------------------------------
    def get_face_mask(self, image_bgr_uint8):
        """Return a float32 (H, W) face mask in [0, 1].

        Ported from ``RetouchDecomposer.get_face_mask_parsing`` — the
        inference stage only needs the face-mask layer. Erodes the parsed
        face region inward and feathers the edge so the dodge & burn never
        spills outside the face. MediaPipe (if available) picks the correct
        connected component when several skin-coloured blobs are detected.
        """
        parsing = self.face_parser.infer(image_bgr_uint8)
        h, w = image_bgr_uint8.shape[:2]
        max_dim = max(h, w)

        # 1. Solid facial region from semantic labels.
        raw_mask = np.isin(parsing, FaceParsing.CORE_FACE_PARTS).astype(np.uint8) * 255

        # 2. Geometric disambiguation: keep the blob nearest the nose tip so
        #    arms / background skin tones are rejected.
        face_center = None
        if self.mp_face_mesh is not None:
            results = self.mp_face_mesh.process(
                cv2.cvtColor(image_bgr_uint8, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                face_center = (int(lm[1].x * w), int(lm[1].y * h))

        if face_center is None:
            final_mask_uint8 = raw_mask
        else:
            num_labels, labels, _, centroids = cv2.connectedComponentsWithStats(raw_mask)
            if num_labels <= 1:
                return np.zeros((h, w), dtype=np.float32)
            distances = [np.linalg.norm(centroids[i] - np.array(face_center))
                         for i in range(1, num_labels)]
            true_face_label = int(np.argmin(distances)) + 1
            final_mask_uint8 = (labels == true_face_label).astype(np.uint8) * 255

        # 3. Erode inward so the edge (jawline / hairline) is excluded.
        shrink_base = int(max_dim * 0.012) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink_base, shrink_base))
        eroded_face = cv2.erode(final_mask_uint8, kernel, iterations=1)

        # 4. Feather the edge to avoid hard seams in the composite.
        blur_size = int(max_dim * 0.08) | 1
        if blur_size % 2 == 0:
            blur_size += 1
        face_mask = cv2.GaussianBlur(
            eroded_face.astype(np.float32) / 255.0, (blur_size, blur_size), 0)

        return face_mask

    # -----------------------------------------------------------------
    # Step 3 — linear superposition of the two layers onto the basic image
    # -----------------------------------------------------------------
    @staticmethod
    def _to_gray_offset(layer_img, target_hw):
        """Convert a layer PIL/np image to a single-channel 0.5-centred offset.

        The high- and low-frequency layers encode change as a neutral-gray
        map (0.5 == 128 == "no change"), so the offset is simply ``v - 0.5``.
        Returns a (H, W) float32 array already resized to ``target_hw``.
        """
        if isinstance(layer_img, Image.Image):
            arr = np.array(layer_img.convert("L"))
        else:
            arr = np.asarray(layer_img)
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2GRAY)
        arr = arr.astype(np.float32) / 255.0

        th, tw = target_hw
        if arr.shape[:2] != (th, tw):
            arr = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_LINEAR)
        return arr - 0.5

    def compose(self, basic_bgr_f32, high_layer, low_layer, face_mask,
                high_coeff=1.0, low_coeff=1.0, need_high_freq=False):
        """Linearly superimpose high/low layers onto the basic image.

            result = raw + (high_offset * high_coeff + low_offset * low_coeff) * mask

        Args:
            basic_bgr_f32: basic image, BGR float32 in [0, 1], shape (H, W, 3).
            high_layer:    high-frequency layer (PIL or np), 0.5-centred.
                           Ignored when ``need_high_freq`` is False.
            low_layer:     low-frequency  layer (PIL or np), 0.5-centred.
            face_mask:     float32 (H, W) in [0, 1].
            need_high_freq: whether to composite the high-frequency layer.
                           Defaults to False — compositing the high-frequency
                           texture residual is usually not very pleasing.
        Returns:
            uint8 BGR image, shape (H, W, 3).
        """
        h, w = basic_bgr_f32.shape[:2]

        # Low-frequency (DNB light/shadow) layer is always composited.
        offset = self._to_gray_offset(low_layer, (h, w)) * low_coeff

        # High-frequency (texture) layer is opt-in.
        if need_high_freq:
            offset = offset + self._to_gray_offset(high_layer, (h, w)) * high_coeff

        offset = offset[..., None]  # (H, W, 1) -> broadcasts over channels

        # Gate by the (feathered) face mask so retouching stays on the face.
        mask = face_mask
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = mask[..., None]

        recon = np.clip(basic_bgr_f32 + offset, 0.0, 1.0)
        result_f = basic_bgr_f32 * (1.0 - mask) + recon * mask
        return (np.clip(result_f, 0.0, 1.0) * 255.0).astype(np.uint8)

    # -----------------------------------------------------------------
    # End-to-end: basic image -> retouched image
    # -----------------------------------------------------------------
    def run_on_image(self, basic_path, prompt, out_dir,
                     precomputed_layers=None, save_intermediate=True,
                     need_high_freq=False, high_coeff=1.0, low_coeff=1.0,
                     **infer_kwargs):
        """Full pipeline for a single basic image.

        Args:
            basic_path: path to the input image (png/jpg).
            prompt: validation prompt for the Qwen model.
            out_dir: directory to write outputs into.
            precomputed_layers: optional (high_path, low_path) tuple to skip
                diffusion and composite pre-rendered layers (debug / no-GPU).
            need_high_freq: composite the high-frequency layer too (default
                False — usually not very pleasing).
            high_coeff / low_coeff: per-layer strength multipliers.
        Returns:
            Path to the saved retouched image.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = Path(basic_path).name
        for tag in ("_basic.png", "_basic.jpg", "_basic.jpeg", ".png", ".jpg", ".jpeg"):
            if prefix.lower().endswith(tag):
                prefix = prefix[: -len(tag)]
                break

        # Basic image: keep a BGR float32 [0,1] copy for compositing.
        basic_bgr = cv2.imread(str(basic_path))
        if basic_bgr is None:
            raise RuntimeError(f"Could not read basic image: {basic_path}")
        basic_bgr_f32 = basic_bgr.astype(np.float32) / 255.0

        # Step 1: layers (infer, or load pre-computed ones).
        if precomputed_layers is not None:
            high_path, low_path = precomputed_layers
            high_layer = Image.open(high_path) if high_path else None
            low_layer = Image.open(low_path)
        else:
            # DEV.md update: feed the model a black & white, S-curve-contrasted
            # version of the basic image so it identifies shadows/highlights
            # better. The original colour image is kept for compositing below.
            bw = convert_to_black_white_with_scaling(
                basic_bgr, contrast_strength=self.contrast_strength)
            if save_intermediate:
                cv2.imwrite(str(out_dir / f"{prefix}_bw_input.png"), bw)
            # Model expects RGBA; replicate the single B&W channel to RGB.
            bw_rgb = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
            image_rgba = Image.fromarray(bw_rgb).convert("RGBA")

            layer_imgs = self.infer_layers(image_rgba, prompt, **infer_kwargs)
            if len(layer_imgs) < 2:
                raise RuntimeError(
                    f"Expected >= 2 layers from the model, got {len(layer_imgs)}.")
            high_layer = layer_imgs[self.HIGH_FREQ_IDX]
            low_layer = layer_imgs[self.LOW_FREQ_IDX]
            if save_intermediate:
                high_layer.save(out_dir / f"{prefix}_layer_high.png")
                low_layer.save(out_dir / f"{prefix}_layer_low.png")

        # Step 2: face mask.
        face_mask = self.get_face_mask(basic_bgr)
        if save_intermediate:
            cv2.imwrite(str(out_dir / f"{prefix}_face_mask.png"),
                        (face_mask * 255.0).astype(np.uint8))

        # Step 3: linear superposition.
        result = self.compose(basic_bgr_f32, high_layer, low_layer, face_mask,
                              need_high_freq=need_high_freq,
                              high_coeff=high_coeff, low_coeff=low_coeff)
        out_path = out_dir / f"{prefix}_dnb_retouch.png"
        cv2.imwrite(str(out_path), result)
        return out_path


# =====================================================================
# Loaders
# =====================================================================
def load_qwen_pipeline(pretrained_path, device,
                       torch_dtype=torch.bfloat16):
    """Load the Qwen-Image-Layered pipeline with the fine-tuned transformer.

    Mirrors ``src/qwen-image-layered-test.py``. Returns the pipeline moved to
    ``device`` with VAE tiling/slicing enabled.
    """
    from diffusers import QwenImageLayeredPipeline, QwenImageTransformer2DModel

    pipeline = QwenImageLayeredPipeline.from_pretrained(
        pretrained_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    pipeline.to(device)
    try:
        pipeline.transformer.set_default_attn_processor()
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not set default attn processor: {e}")
    pipeline.vae.enable_tiling()
    pipeline.vae.enable_slicing()
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def build_face_mesh():
    """Optional MediaPipe FaceMesh for connected-component disambiguation."""
    try:
        import mediapipe as mp
        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    except Exception as e:  # noqa: BLE001
        print(f"Warning: MediaPipe unavailable, face-mask refinement disabled: {e}")
        return None




