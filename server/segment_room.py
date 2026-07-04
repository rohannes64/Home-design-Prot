"""
Arteffects Room Segmentation
Uses SegFormer-B2 (pretrained on ADE20K) for semantic segmentation.
First run downloads ~85MB model from HuggingFace (one time only).

Install:
    pip install torch torchvision transformers opencv-python Pillow numpy

Run:
    python segment_room.py room.jpg
    python segment_room.py C:/photos/              # whole folder
"""

import cv2
import numpy as np
import sys
import os
import json
import argparse
from pathlib import Path
from PIL import Image

# ── Lazy imports so startup errors are clear ───────────────────────────────
try:
    import torch
    from transformers import (
        SegformerForSemanticSegmentation,
        SegformerImageProcessor,
    )
except ImportError:
    print("Missing dependencies. Run:")
    print("  pip install torch torchvision transformers opencv-python Pillow numpy")
    sys.exit(1)


# ── ADE20K label IDs we care about ────────────────────────────────────────
# Full list: https://huggingface.co/datasets/huggingface/label-files/blob/main/ade20k-id2label.json
ADE20K_ZONES = {
    # id  : (zone_name,  BGR_color_for_overlay)
    0:   ("wall",     (60,  160,  60)),
    3:   ("floor",    (40,  90,  210)),
    5:   ("ceiling",  (160, 100,  40)),
    8:   ("window",   (200, 200,  60)),
    14:  ("door",     (200, 130,  60)),
    43:  ("column",   (0,   220, 220)),
    53:  ("stairs",   (180,  60, 180)),
    # Everything else → "other" (ignored / transparent)
}

ZONE_PRIORITY = ["ceiling", "wall", "floor", "column", "door", "window", "stairs"]

MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"


# ── Load model (cached after first download) ──────────────────────────────
def load_model():
    try:
        # Try loading from local cache first to prevent slow network version checks
        processor = SegformerImageProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
        model     = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME, local_files_only=True)
    except Exception:
        # Fallback to downloading if not cached yet
        processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
        model     = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device


# ── Run inference ─────────────────────────────────────────────────────────
def run_segformer(img_bgr, processor, model, device):
    """
    Returns a (H, W) numpy array where each pixel value is an ADE20K label id.
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    inputs = processor(images=pil_img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Upsample logits back to original image size
    logits = outputs.logits                  # (1, num_labels, H/4, W/4)
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )
    pred = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
    return pred


# ── Build colored overlay from predictions ────────────────────────────────
def build_overlay(img_bgr, pred):
    h, w = img_bgr.shape[:2]
    color_layer = np.zeros((h, w, 3), dtype=np.uint8)
    zone_masks  = {}

    for label_id, (zone_name, color) in ADE20K_ZONES.items():
        mask = (pred == label_id).astype(np.uint8) * 255
        if mask.sum() == 0:
            continue
        color_layer[mask > 0] = color
        zone_masks[zone_name] = mask

    overlay = cv2.addWeighted(img_bgr.copy(), 0.55, color_layer, 0.45, 0)
    return overlay, zone_masks


# ── Draw clean contour lines between zones ────────────────────────────────
def draw_zone_borders(overlay, zone_masks):
    for zone_name, mask in zone_masks.items():
        _, _, color = None, None, None
        for lid, (zn, col) in ADE20K_ZONES.items():
            if zn == zone_name:
                color = col
                break
        if color is None:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

    return overlay


# ── Add zone labels at centroid of each zone ──────────────────────────────
def draw_labels(overlay, zone_masks, img_shape):
    h, w = img_shape[:2]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    font_s = min(1.1, w / 900)

    for zone_name in ZONE_PRIORITY:
        if zone_name not in zone_masks:
            continue
        mask = zone_masks[zone_name]

        # Find centroid
        M = cv2.moments(mask)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Coverage %
        pct = np.sum(mask > 0) / (h * w) * 100

        label = f"{zone_name.upper()}  {pct:.0f}%"

        # Get zone color
        zone_color = (255, 255, 255)
        for lid, (zn, col) in ADE20K_ZONES.items():
            if zn == zone_name:
                zone_color = col
                break

        cv2.putText(overlay, label, (cx - 60, cy), font, font_s, (0, 0, 0), 5)
        cv2.putText(overlay, label, (cx - 60, cy), font, font_s, zone_color,  2)

    return overlay


# ── Build individual mask strips (stacked vertically) ────────────────────
def build_mask_strips(img_bgr, zone_masks):
    h, w = img_bgr.shape[:2]
    strip_h = 300
    strip_w = int(w * strip_h / h)
    font    = cv2.FONT_HERSHEY_SIMPLEX

    strips = []
    for zone_name in ZONE_PRIORITY:
        if zone_name not in zone_masks:
            continue
        mask = zone_masks[zone_name]

        # Apply mask to original image
        masked = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        strip  = cv2.resize(masked, (strip_w, strip_h))

        # Border color
        border_color = (255, 255, 255)
        for lid, (zn, col) in ADE20K_ZONES.items():
            if zn == zone_name:
                border_color = col
                break

        cv2.rectangle(strip, (0, 0), (strip_w - 1, strip_h - 1), border_color, 4)

        pct = np.sum(mask > 0) / (h * w) * 100
        cv2.putText(strip, zone_name.upper(), (10, 32), font, 0.9, (0,0,0), 4)
        cv2.putText(strip, zone_name.upper(), (10, 32), font, 0.9, border_color, 2)
        cv2.putText(strip, f"{pct:.1f}% of image", (10, strip_h - 10),
                    font, 0.5, (200, 200, 200), 1)

        strips.append(strip)

    if not strips:
        return None

    max_w = max(s.shape[1] for s in strips)
    padded = [
        cv2.copyMakeBorder(s, 0, 6, 0, max_w - s.shape[1],
                           cv2.BORDER_CONSTANT, value=(15, 15, 15))
        for s in strips
    ]
    return np.vstack(padded)


# ── Indian lighting enhancement ───────────────────────────────────────────
def enhance_lighting(img):
    """Normalize warm/dim Indian home lighting before inference"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ── Print JSON summary (legacy, human-readable) ──────────────────────────
def print_json(zone_masks, img_shape):
    h, w = img_shape[:2]
    print("\nJSON output (for API use):")
    print("{")
    for zone_name in ZONE_PRIORITY:
        if zone_name not in zone_masks:
            continue
        mask = zone_masks[zone_name]
        rows = np.where(np.any(mask > 0, axis=1))[0]
        cols = np.where(np.any(mask > 0, axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            continue
        pct = np.sum(mask > 0) / (h * w)
        print(f'  "{zone_name}": {{')
        print(f'    "y_start": {int(rows[0])}, "y_end": {int(rows[-1])},')
        print(f'    "x_start": {int(cols[0])}, "x_end": {int(cols[-1])},')
        print(f'    "coverage_pct": {pct:.3f}')
        print(f'  }},')
    print("}")


# ── JSON-only segmentation (for Node.js API bridge) ──────────────────────
def segment_json_only(image_path, processor, model, device):
    """
    Fast segmentation that outputs ONLY machine-parseable JSON to stdout.
    Skips overlay generation, labels, mask strips — purely for API consumption.
    """
    img = cv2.imread(image_path)
    if img is None:
        json.dump({"error": f"Could not load: {image_path}"}, sys.stdout)
        return

    h, w = img.shape[:2]

    # Enhance and run inference
    enhanced = enhance_lighting(img)
    pred = run_segformer(enhanced, processor, model, device)

    # Build zone masks (lightweight — no overlay)
    zones = []
    for label_id, (zone_name, _color) in ADE20K_ZONES.items():
        mask = (pred == label_id).astype(np.uint8) * 255
        pixel_count = int(np.sum(mask > 0))
        if pixel_count == 0:
            continue

        rows = np.where(np.any(mask > 0, axis=1))[0]
        cols = np.where(np.any(mask > 0, axis=0))[0]
        coverage = pixel_count / (h * w)

        zones.append({
            "name": zone_name,
            "coverage": round(coverage, 4),
            "pixel_count": pixel_count,
            "bbox": {
                "x": int(cols[0]),
                "y": int(rows[0]),
                "w": int(cols[-1] - cols[0]),
                "h": int(rows[-1] - rows[0])
            }
        })

    # Sort by coverage descending
    zones.sort(key=lambda z: z["coverage"], reverse=True)

    result = {
        "zones": zones,
        "image_size": {"width": w, "height": h}
    }

    # Output ONLY valid JSON to stdout (no other prints)
    json.dump(result, sys.stdout)
    sys.stdout.flush()


def tile_image(texture, target_shape):
    h, w = target_shape[:2]
    th, tw = texture.shape[:2]
    repeat_y = (h + th - 1) // th
    repeat_x = (w + tw - 1) // tw
    tiled = np.tile(texture, (repeat_y, repeat_x, 1))
    return tiled[:h, :w]


# ── Main segmentation function ────────────────────────────────────────────
def segment_room(image_path, output_dir, processor, model, device, target_zone=None, mask_output_path=None, texture_path=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not load: {image_path}")
        return

    print(f"\n{'='*52}")
    print(f"Image : {os.path.basename(image_path)}")
    h, w = img.shape[:2]
    print(f"Size  : {w} x {h}")

    # 1. Enhance for Indian lighting conditions
    enhanced = enhance_lighting(img)

    # 2. Run SegFormer
    print("  Running SegFormer inference...")
    pred = run_segformer(enhanced, processor, model, device)

    # 3. Build outputs
    overlay, zone_masks = build_overlay(img, pred)

    # Save target zone binary mask if requested
    if target_zone and mask_output_path:
        tz_lower = target_zone.lower()
        mask = zone_masks.get(tz_lower)
        if mask is not None:
            # Ensure parent directories exist
            mask_dir = os.path.dirname(mask_output_path)
            if mask_dir:
                os.makedirs(mask_dir, exist_ok=True)
            
            if texture_path and os.path.exists(texture_path):
                # Composite mode: tile texture onto room photo within the mask
                print(f"  Tiling texture from {texture_path} onto zone '{target_zone}'...")
                texture = cv2.imread(texture_path)
                if texture is not None:
                    tiled_texture = tile_image(texture, img.shape)
                    if len(tiled_texture.shape) == 2:
                        tiled_texture = cv2.cvtColor(tiled_texture, cv2.COLOR_GRAY2BGR)
                    
                    # Create float mask for blending (0.0 to 1.0)
                    alpha = mask.astype(np.float32) / 255.0
                    alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1)
                    
                    # Blend: composited = texture * alpha + room * (1 - alpha)
                    composited = (tiled_texture.astype(np.float32) * alpha_3ch + 
                                  img.astype(np.float32) * (1.0 - alpha_3ch))
                    composited = np.clip(composited, 0, 255).astype(np.uint8)
                    
                    cv2.imwrite(mask_output_path, composited, [cv2.IMWRITE_JPEG_QUALITY, 93])
                    print(f"  Saved composited room -> {mask_output_path}")
                else:
                    print(f"  WARNING: Could not load texture from '{texture_path}'. Falling back to mask-only output.")
                    rgba_mask = cv2.merge([mask, mask, mask, mask])
                    cv2.imwrite(mask_output_path, rgba_mask)
            else:
                # Mask-only mode: output RGBA transparent PNG
                rgba_mask = cv2.merge([mask, mask, mask, mask])
                cv2.imwrite(mask_output_path, rgba_mask)
                print(f"  Saved binary mask for zone '{target_zone}' -> {mask_output_path}")
        else:
            # Output an empty black mask if requested zone is not found
            print(f"  WARNING: Zone '{target_zone}' not detected in this image. Generating empty black mask.")
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            mask_dir = os.path.dirname(mask_output_path)
            if mask_dir:
                os.makedirs(mask_dir, exist_ok=True)
            if texture_path:
                # If texture was requested but zone not found, output original room
                cv2.imwrite(mask_output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 93])
                print(f"  Saved original room (zone not found) -> {mask_output_path}")
            else:
                empty_rgba = cv2.merge([empty_mask, empty_mask, empty_mask, empty_mask])
                cv2.imwrite(mask_output_path, empty_rgba)
                print(f"  Saved empty binary mask -> {mask_output_path}")

    overlay = draw_zone_borders(overlay, zone_masks)
    overlay = draw_labels(overlay, zone_masks, img.shape)
    mask_strips = build_mask_strips(img, zone_masks)

    # 4. Print detected zones
    print(f"  Zones detected: {list(zone_masks.keys())}")

    # 5. Save
    stem     = Path(image_path).stem
    out_seg  = os.path.join(output_dir, f"{stem}_segmented.jpg")
    out_mask = os.path.join(output_dir, f"{stem}_masks.jpg")

    cv2.imwrite(out_seg, overlay, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f"  Saved -> {out_seg}")

    if mask_strips is not None:
        cv2.imwrite(out_mask, mask_strips, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(f"  Saved -> {out_mask}")

    # 6. JSON
    print_json(zone_masks, img.shape)


# ── Entry point ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Arteffects Room Segmentation")
    parser.add_argument("input_path", help="Path to room image file or directory")
    parser.add_argument("--zone", help="Target zone to generate binary mask for (e.g. floor, wall, ceiling)")
    parser.add_argument("--output_mask", help="Path to save the black-and-white binary mask PNG")
    parser.add_argument("--output_dir", default="segmented_output", help="Directory to save default overlay images")
    parser.add_argument("--texture_path", help="Path to product texture image to tile and mask")
    parser.add_argument("--json_only", action="store_true",
                        help="Output ONLY machine-parseable JSON to stdout (no visuals). For API use.")
    
    args = parser.parse_args()
    input_path = args.input_path

    if not os.path.exists(input_path):
        if args.json_only:
            json.dump({"error": f"File not found: {input_path}"}, sys.stdout)
        else:
            print(f"File not found: {input_path}")
        sys.exit(1)

    # Load model once (suppress prints in json_only mode)
    if args.json_only:
        # Redirect stderr to suppress model loading messages
        import io
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        old_stdout_write = print  # We'll use json.dump directly
    
    processor, model, device = load_model()

    if args.json_only:
        sys.stderr = old_stderr
        # JSON-only mode: single image, clean JSON output
        if os.path.isdir(input_path):
            json.dump({"error": "--json_only does not support directories"}, sys.stdout)
            sys.exit(1)
        segment_json_only(input_path, processor, model, device)
        return

    # Normal mode (visual overlays)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(input_path):
        exts   = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        images = sorted([
            f for f in Path(input_path).iterdir()
            if f.suffix.lower() in exts
        ])
        if not images:
            print(f"No images found in {input_path}")
            sys.exit(1)
        print(f"\nFound {len(images)} image(s) — processing all")
        for p in images:
            segment_room(str(p), output_dir, processor, model, device, args.zone, args.output_mask, args.texture_path)
    else:
        segment_room(input_path, output_dir, processor, model, device, args.zone, args.output_mask, args.texture_path)

    print(f"\nAll done. Results in: {os.path.abspath(output_dir)}/")


if __name__ == "__main__":
    main()
