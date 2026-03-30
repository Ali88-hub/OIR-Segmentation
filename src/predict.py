"""Prediction pipeline for retinal segmentation.

Usage:
    # Single image
    python -m src.predict --checkpoint best_model.pth --input image.png --output output/

    # Directory of images
    python -m src.predict --checkpoint best_model.pth --input images/ --output output/

    # With TTA and custom threshold
    python -m src.predict --checkpoint best_model.pth --input images/ --output output/ --tta --threshold 0.45
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.measure import label as sk_label, regionprops
from torch.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import Config
from src.model import build_model


MASK_COLORS = {
    "nv": (0.7, 0.0, 1.0),          # purple (matches app.py)
    "vo": (0.0, 0.5, 1.0),         # blue
    "retina": (0.0, 0.8, 0.0), # green
}


def load_model(checkpoint_path, config, device):
    """Load model from checkpoint, overriding architecture config from saved state."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Override architecture fields from checkpoint if available
    if "config" in ckpt:
        saved = ckpt["config"]
        config.image_size = tuple(saved.get("image_size", config.image_size))
        config.encoder_name = saved.get("encoder_name", config.encoder_name)
        config.decoder_attention = saved.get("decoder_attention", config.decoder_attention)
        config.num_classes = saved.get("num_classes", config.num_classes)
        config.mask_names = tuple(saved.get("mask_names", config.mask_names))

    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


MAX_INPUT_SIZE = 1024  # images larger than this are downscaled before inference


def get_preprocess(config):
    """Validation-style preprocessing: resize + normalize."""
    return A.Compose([
        A.Resize(config.image_size[0], config.image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def resize_to_max(image_np, max_side=MAX_INPUT_SIZE):
    """Downscale image so its longest side <= max_side, preserving aspect ratio.

    Returns:
        resized_np: downscaled uint8 image
        scale: float, resized/original (same for both axes)
    """
    h, w = image_np.shape[:2]
    if h <= max_side and w <= max_side:
        return image_np, 1.0
    scale = max_side / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = np.array(Image.fromarray(image_np).resize((new_w, new_h), Image.LANCZOS))
    print(f"  Resized {w}x{h} -> {new_w}x{new_h} (scale={scale:.4f})")
    return resized, scale


def predict_single(model, image_np, preprocess, device, config, tta=False, threshold=0.5):
    """Run inference on a single image.

    Args:
        model: trained model in eval mode
        image_np: HxWx3 uint8 numpy array (RGB)
        preprocess: albumentations transform
        device: torch device
        config: Config object
        tta: if True, average predictions over flips
        threshold: binarization threshold

    Returns:
        masks_prob: [num_classes, H, W] float32 probabilities (original resolution)
        masks_binary: [num_classes, H, W] uint8 binary masks (original resolution)
    """
    orig_h, orig_w = image_np.shape[:2]

    def _infer(img_np):
        t = preprocess(image=img_np)["image"].unsqueeze(0).to(device)
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(t)
        return logits.squeeze(0).detach().cpu()

    logits = _infer(image_np)

    if tta:
        # Horizontal flip
        l_hflip = _infer(image_np[:, ::-1].copy())
        l_hflip = torch.flip(l_hflip, dims=[2])
        # Vertical flip
        l_vflip = _infer(image_np[::-1, :].copy())
        l_vflip = torch.flip(l_vflip, dims=[1])
        # Both flips
        l_hvflip = _infer(image_np[::-1, ::-1].copy())
        l_hvflip = torch.flip(l_hvflip, dims=[1, 2])

        logits = (logits + l_hflip + l_vflip + l_hvflip) / 4.0

    probs = torch.sigmoid(logits)

    # Resize probabilities back to original resolution
    probs_np = probs.numpy()
    masks_prob = np.zeros((config.num_classes, orig_h, orig_w), dtype=np.float32)
    for i in range(config.num_classes):
        resized = np.array(
            Image.fromarray(probs_np[i]).resize((orig_w, orig_h), Image.BILINEAR)
        )
        masks_prob[i] = resized

    masks_binary = (masks_prob > threshold).astype(np.uint8)
    return masks_prob, masks_binary


def predict_tiled(model, image_np, preprocess, device, config, tta=False, threshold=0.5,
                  tile_size=512, overlap=128):
    """Tiled inference for large images with overlap blending.

    Splits the image into overlapping tiles, runs inference on each, then
    stitches predictions back using a linear blend in the overlap zones.
    """
    orig_h, orig_w = image_np.shape[:2]
    num_classes = config.num_classes
    stride = tile_size - overlap

    acc = np.zeros((num_classes, orig_h, orig_w), dtype=np.float64)
    weight = np.zeros((orig_h, orig_w), dtype=np.float64)

    # 1-D linear ramp for blending: 0→1 over overlap, 1 in center, 1→0 over overlap
    def make_blend_1d(size):
        w = np.ones(size, dtype=np.float64)
        ramp = np.linspace(0, 1, overlap, endpoint=False)
        w[:overlap] = ramp
        w[size - overlap:] = ramp[::-1]
        return w

    blend_h = make_blend_1d(tile_size)
    blend_w = make_blend_1d(tile_size)
    blend_2d = np.outer(blend_h, blend_w)  # (tile_size, tile_size)

    # Build tile grid (top-left corners)
    ys = list(range(0, orig_h - tile_size, stride)) + [orig_h - tile_size]
    xs = list(range(0, orig_w - tile_size, stride)) + [orig_w - tile_size]
    ys = sorted(set(max(0, y) for y in ys))
    xs = sorted(set(max(0, x) for x in xs))

    total = len(ys) * len(xs)
    print(f"  Tiled inference: {orig_h}x{orig_w} -> {len(ys)}x{len(xs)} = {total} tiles")

    def _infer_tile(tile_np):
        t = preprocess(image=tile_np)["image"].unsqueeze(0).to(device)
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(t)
        return logits.squeeze(0).detach().cpu().numpy()  # (C, tile_size, tile_size)

    count = 0
    for y in ys:
        for x in xs:
            tile = image_np[y:y + tile_size, x:x + tile_size]
            # Pad if tile is smaller than expected (edge case)
            th, tw = tile.shape[:2]
            if th < tile_size or tw < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:th, :tw] = tile
                tile = padded

            logits_tile = _infer_tile(tile)

            if tta:
                l_hflip = _infer_tile(tile[:, ::-1].copy())
                l_hflip = l_hflip[:, :, ::-1]
                l_vflip = _infer_tile(tile[::-1, :].copy())
                l_vflip = l_vflip[:, ::-1, :]
                l_hvflip = _infer_tile(tile[::-1, ::-1].copy())
                l_hvflip = l_hvflip[:, ::-1, ::-1]
                logits_tile = (logits_tile + l_hflip + l_vflip + l_hvflip) / 4.0

            # Accumulate with blend weights
            actual_h = min(tile_size, orig_h - y)
            actual_w = min(tile_size, orig_w - x)
            b = blend_2d[:actual_h, :actual_w]
            acc[:, y:y + actual_h, x:x + actual_w] += logits_tile[:, :actual_h, :actual_w] * b
            weight[y:y + actual_h, x:x + actual_w] += b

            count += 1
            if count % 50 == 0 or count == total:
                print(f"    {count}/{total} tiles done")

    # Normalize by accumulated weights, then sigmoid to get probabilities
    weight = np.maximum(weight, 1e-8)
    masks_logits = (acc / weight).astype(np.float32)
    masks_prob = (1.0 / (1.0 + np.exp(-masks_logits))).astype(np.float32)
    masks_binary = (masks_prob > threshold).astype(np.uint8)
    return masks_prob, masks_binary


# ── Post-processing ───────────────────────────────────────────────────────────

def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """Fill holes then keep only the largest connected component."""
    filled = ndimage.binary_fill_holes(mask).astype(np.uint8)
    labeled, n = ndimage.label(filled)
    if n == 0:
        return filled
    largest = int(np.argmax(ndimage.sum(filled, labeled, range(1, n + 1)))) + 1
    return (labeled == largest).astype(np.uint8)


def postprocess_vo(mask: np.ndarray, close_radius: int = 15) -> np.ndarray:
    """Aggressive VO post-processing: close gaps, fill holes, keep largest component."""
    struct = ndimage.generate_binary_structure(2, 1)
    struct = ndimage.iterate_structure(struct, close_radius)
    closed = ndimage.binary_closing(mask.astype(bool), structure=struct)
    filled = ndimage.binary_fill_holes(closed).astype(np.uint8)
    labeled, n = ndimage.label(filled)
    if n == 0:
        return filled
    largest = int(np.argmax(ndimage.sum(filled, labeled, range(1, n + 1)))) + 1
    return (labeled == largest).astype(np.uint8)


def postprocess_nv(
    nv_mask: np.ndarray,
    vo_mask: np.ndarray,
    vessel_mask: np.ndarray | None = None,
    outside_px: int = 520,
    inside_px: int = 260,
    min_area: int = 150,
    max_eccentricity: float = 0.985,
    vessel_suppression: bool = True,
) -> np.ndarray:
    """Post-process NV mask to reduce false positives from normal vessels.

    Three stages:
    A. VO-boundary spatial masking — zero out NV far from the VO edge
    B. Vessel mask suppression — zero out NV overlapping known vessels
    C. Morphological filtering — remove elongated/tiny connected components
    """
    result = nv_mask.copy()

    # A. VO-boundary spatial masking
    vo_bool = vo_mask.astype(bool)
    if vo_bool.any():
        # Distance from each non-VO pixel to nearest VO pixel
        dist_outside = distance_transform_edt(~vo_bool)
        # Distance from each VO pixel to nearest non-VO pixel (VO interior depth)
        dist_inside = distance_transform_edt(vo_bool)
        # Boundary zone = within outside_px of VO edge (outside) and within inside_px (inside)
        boundary_zone = (dist_outside <= outside_px) & (dist_inside <= inside_px)
        result = result & boundary_zone.astype(np.uint8)

    # B. Vessel mask suppression
    if vessel_suppression and vessel_mask is not None:
        if vessel_mask.shape != result.shape:
            vessel_mask = np.array(Image.fromarray(vessel_mask).resize(
                (result.shape[1], result.shape[0]), Image.NEAREST))
        result = result & (~vessel_mask.astype(bool)).astype(np.uint8)

    # C. Morphological component filtering
    if result.any():
        labeled = sk_label(result, connectivity=2)
        for region in regionprops(labeled):
            if region.area < min_area or region.eccentricity > max_eccentricity:
                result[labeled == region.label] = 0

    return result


def postprocess_all(
    masks_binary: np.ndarray,
    mask_names: tuple,
    vessel_mask: np.ndarray | None = None,
    config=None,
) -> np.ndarray:
    """Apply class-specific post-processing to all masks.

    Order matters: VO is cleaned first so NV boundary masking uses a clean VO.

    Args:
        masks_binary: [num_classes, H, W] uint8 binary masks
        mask_names: tuple of class names, e.g. ("nv", "vo", "retina")
        vessel_mask: optional [H, W] uint8 binary vessel mask
        config: Config object (uses defaults if None)
    """
    from src.config import Config
    if config is None:
        config = Config()

    result = masks_binary.copy()
    names = list(mask_names)

    # 1. VO post-processing (must be first — NV needs clean VO)
    if "vo" in names:
        result[names.index("vo")] = postprocess_vo(result[names.index("vo")])

    # 2. Retina post-processing
    if "retina" in names:
        result[names.index("retina")] = postprocess_mask(result[names.index("retina")])

    # 3. NV post-processing (uses cleaned VO mask)
    if "nv" in names and "vo" in names:
        nv_idx = names.index("nv")
        vo_idx = names.index("vo")
        result[nv_idx] = postprocess_nv(
            result[nv_idx],
            result[vo_idx],
            vessel_mask=vessel_mask,
            outside_px=config.nv_outside_px,
            inside_px=config.nv_inside_px,
            min_area=config.nv_min_component_area,
            max_eccentricity=config.nv_max_eccentricity,
            vessel_suppression=config.nv_vessel_suppression,
        )

    return result


# ── Vessel mask loading ───────────────────────────────────────────────────────

_manifest_cache: dict[str, pd.DataFrame] = {}


def load_vessel_mask(
    image_stem: str,
    manifest_path: str,
    vessel_mask_root: str = "data/Training data",
    vessel_mask_fallback: str = "data/vessels mask",
) -> np.ndarray | None:
    """Load a ground-truth vessel mask by image stem, if available.

    Tries manifest vessel_mask_path first, then falls back to the
    loose vessel mask folder (data/vessels mask/) by stem name.

    Returns [H, W] uint8 binary mask, or None if not found.
    """
    if manifest_path not in _manifest_cache:
        try:
            _manifest_cache[manifest_path] = pd.read_csv(manifest_path)
        except FileNotFoundError:
            return None
    df = _manifest_cache[manifest_path]

    rows = df[df["stem"] == image_stem]
    if rows.empty:
        return None

    # Try 1: manifest vessel_mask_path column
    row = rows.iloc[0]
    vessel_path = row.get("vessel_mask_path", "")
    if vessel_path and not (isinstance(vessel_path, float) and np.isnan(vessel_path)):
        full_path = Path(vessel_mask_root) / Path(str(vessel_path).replace("\\", "/"))
        if full_path.exists():
            mask = np.array(Image.open(str(full_path)).convert("L"))
            return (mask > 127).astype(np.uint8)

    # Try 2: fallback folder by stem name (.jpg then .png)
    fallback_dir = Path(vessel_mask_fallback)
    if fallback_dir.is_dir():
        for ext in (".jpg", ".png", ".JPG", ".PNG"):
            fallback_path = fallback_dir / f"{image_stem}{ext}"
            if fallback_path.exists():
                mask = np.array(Image.open(str(fallback_path)).convert("L"))
                return (mask > 127).astype(np.uint8)

    return None


def save_masks(masks_binary, mask_names, output_dir, stem):
    """Save individual binary masks as PNGs."""
    for i, name in enumerate(mask_names):
        mask_img = Image.fromarray(masks_binary[i] * 255)
        mask_img.save(os.path.join(output_dir, f"{stem}_{name}.png"))


def save_overlay_large(image_np, masks_binary, masks_prob, mask_names, output_dir, stem,
                        max_side=4096):
    """Save 4-panel overlay for large images using PIL (matches save_overlay layout)."""
    from PIL import ImageDraw, ImageFont

    orig_h, orig_w = image_np.shape[:2]

    # Downscale each panel so longest side <= max_side / 2 (4 panels fit in ~2x width)
    panel_max = max_side // 2
    scale = min(panel_max / orig_w, panel_max / orig_h, 1.0)
    pw = int(orig_w * scale)
    ph = int(orig_h * scale)

    base = Image.fromarray(image_np).resize((pw, ph), Image.LANCZOS)

    mask_colors_rgba = {
        "nv":         (178,   0, 255),
        "vo":         (  0, 128, 255),
        "retina":     (  0, 204,   0),
    }

    title_h = 30  # pixels for title bar
    panel_names = ["Input"] + list(mask_names)
    n_panels = len(panel_names)
    canvas_w = pw * n_panels
    canvas_h = ph + title_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # Panel 0: Input (unmodified)
    canvas.paste(base, (0, title_h))
    draw.text((pw // 2, title_h // 2), "Input", fill=(255, 255, 255), anchor="mm")

    # Panels 1–3: one mask each
    for i, name in enumerate(mask_names):
        panel = base.copy().convert("RGBA")
        color = mask_colors_rgba.get(name, (255, 255, 0))

        mask_small = np.array(
            Image.fromarray(masks_binary[i].astype(np.uint8) * 255).resize((pw, ph), Image.NEAREST)
        )
        color_f = tuple(c / 255.0 for c in color[:3])
        base_np = np.array(base.convert("RGB")).astype(np.float32) / 255.0
        alpha = (mask_small > 0).astype(np.float32) * 0.55
        blended = base_np.copy()
        for c, cv in enumerate(color_f):
            blended[..., c] = base_np[..., c] * (1 - alpha) + cv * alpha
        blended_uint8 = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
        panel = Image.fromarray(blended_uint8)
        x_offset = (i + 1) * pw
        canvas.paste(panel.convert("RGB"), (x_offset, title_h))
        draw.text((x_offset + pw // 2, title_h // 2), name,
                  fill=tuple(color), anchor="mm")

    out_path = os.path.join(output_dir, f"{stem}_overlay.png")
    canvas.save(out_path)
    print("  Overlay saved -> " + out_path + " (" + str(canvas_w) + "x" + str(canvas_h) + ")")


def save_overlay(image_np, masks_binary, masks_prob, mask_names, output_dir, stem):
    """Save a visualization overlay with original image and colored masks."""
    fig, axes = plt.subplots(1, 1 + len(mask_names), figsize=(5 * (1 + len(mask_names)), 5))

    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title("Input")
    axes[0].axis("off")

    # Individual mask predictions
    for i, name in enumerate(mask_names):
        color = MASK_COLORS.get(name, (1, 1, 0))
        alpha = masks_binary[i].astype(np.float32) * 0.55
        base = image_np.astype(np.float32) / 255.0
        blended = base.copy()
        for c, cv in enumerate(color):
            blended[..., c] = base[..., c] * (1 - alpha) + cv * alpha
        blended = np.clip(blended, 0, 1)

        axes[i + 1].imshow(blended)
        axes[i + 1].set_title(f"{name}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{stem}_overlay.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def predict_directory(model, input_dir, output_dir, config, device, tta=False, threshold=0.5):
    """Run prediction on all images in a directory."""
    preprocess = get_preprocess(config)
    input_path = Path(input_dir)
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_files = sorted(
        f for f in input_path.iterdir()
        if f.suffix.lower() in extensions and f.is_file()
    )

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, "masks")
    overlay_dir = os.path.join(output_dir, "overlays")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    Image.MAX_IMAGE_PIXELS = None
    print(f"Predicting {len(image_files)} images...")
    for i, img_path in enumerate(image_files):
        image_np = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = image_np.shape[:2]
        masks_prob, masks_binary = predict_single(
            model, image_np, preprocess, device, config, tta=tta, threshold=threshold
        )

        stem = img_path.stem
        save_masks(masks_binary, config.mask_names, mask_dir, stem)
        if orig_h > MAX_INPUT_SIZE or orig_w > MAX_INPUT_SIZE:
            save_overlay_large(image_np, masks_binary, masks_prob, config.mask_names, overlay_dir, stem)
        else:
            save_overlay(image_np, masks_binary, masks_prob, config.mask_names, overlay_dir, stem)

        print(f"  [{i+1}/{len(image_files)}] {img_path.name}")

    print(f"Done. Masks saved to {mask_dir}, overlays to {overlay_dir}")


def main():
    parser = argparse.ArgumentParser(description="Retinal segmentation prediction")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--input", required=True, help="Path to image or directory")
    parser.add_argument("--output", default="predictions", help="Output directory")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold")
    parser.add_argument("--device", default=None, help="Device (auto-detected if not set)")
    parser.add_argument("--no-attention", action="store_true",
                        help="Disable decoder attention (for checkpoints trained without scSE)")
    args = parser.parse_args()

    config = Config()
    if args.no_attention:
        config.decoder_attention = None

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    model = load_model(args.checkpoint, config, device)

    input_path = Path(args.input)

    if input_path.is_file():
        preprocess = get_preprocess(config)
        os.makedirs(args.output, exist_ok=True)
        Image.MAX_IMAGE_PIXELS = None
        image_np = np.array(Image.open(input_path).convert("RGB"))
        orig_h, orig_w = image_np.shape[:2]
        masks_prob, masks_binary = predict_single(
            model, image_np, preprocess, device, config,
            tta=args.tta, threshold=args.threshold
        )
        stem = input_path.stem
        mask_dir = os.path.join(args.output, "masks")
        overlay_dir = os.path.join(args.output, "overlays")
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)
        save_masks(masks_binary, config.mask_names, mask_dir, stem)
        if orig_h > MAX_INPUT_SIZE or orig_w > MAX_INPUT_SIZE:
            save_overlay_large(image_np, masks_binary, masks_prob, config.mask_names, overlay_dir, stem)
        else:
            save_overlay(image_np, masks_binary, masks_prob, config.mask_names, overlay_dir, stem)
        print(f"Saved to {args.output}")
    elif input_path.is_dir():
        predict_directory(
            model, args.input, args.output, config, device,
            tta=args.tta, threshold=args.threshold
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()
