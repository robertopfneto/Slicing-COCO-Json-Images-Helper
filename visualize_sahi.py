#!/usr/bin/env python3
"""
Visualizador das fronteiras/recortes gerados pelo slicing (SAHI/ASAHI).

Para cada imagem original ele desenha, sobre a própria imagem, os tiles que
foram realmente criados (parsing dos nomes *_tile_{x}_{y}.jpg) e, opcionalmente,
as anotações originais. Útil para conferir se a malha 640x640 cobre bem a
imagem e como ficaram os recortes adaptativos.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.coco import CocoDataset, CocoAnnotation  # noqa: E402
from src.utils.visualization import BoundingBoxVisualizer  # noqa: E402

VALID_MODES = {"sahi", "asahi"}
_TILE_PATTERN = re.compile(r"_tile_(\d+)_([0-9]+)")


def _resolve_annotation_path(tiled_root: str, mode: str, fold: int, split: str) -> str:
    """
    Resolve annotation path for tiled data.
    - Flat layout: <tiled_root>/_annotations.coco.json
    - SAHI legacy: <tiled_root>/<split>/_annotations.coco.json
    - ASAHI fold:  <tiled_root>/fold_<n>/<split>/_annotations.coco.json
    """
    flat = os.path.join(tiled_root, "_annotations.coco.json")
    if os.path.exists(flat):
        return flat

    if mode == "sahi":
        return os.path.join(tiled_root, split, "_annotations.coco.json")
    return os.path.join(tiled_root, f"fold_{fold}", split, "_annotations.coco.json")


def _resolve_image_dir(tiled_root: str, mode: str, fold: int, split: str) -> str:
    """
    Resolve image directory for tiled data (flat or split folders).
    """
    if os.path.isdir(tiled_root):
        # Flat layout already holds the images
        if any(fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")) for fname in os.listdir(tiled_root)):
            return tiled_root

    if mode == "sahi":
        return os.path.join(tiled_root, split)
    return os.path.join(tiled_root, f"fold_{fold}", split)


def _resolve_original_annotation_path(original_root: str) -> str:
    """Handle flat datasets (dataset/_annotations.coco.json) without train/."""
    flat = os.path.join(original_root, "_annotations.coco.json")
    if os.path.exists(flat):
        return flat
    return os.path.join(original_root, "train", "_annotations.coco.json")


def _resolve_original_image_path(original_root: str, filename: str) -> str:
    flat = os.path.join(original_root, filename)
    if os.path.exists(flat):
        return flat
    return os.path.join(original_root, "train", filename)


def _load_datasets(original_path: str, tiled_path: str) -> Tuple[CocoDataset, CocoDataset]:
    original_dataset = CocoDataset.from_json(original_path)
    tiled_dataset = CocoDataset.from_json(tiled_path)
    return original_dataset, tiled_dataset


def _collect_tiles_for_image(base_name: str, tiled_dataset: CocoDataset) -> List[Dict[str, int]]:
    """Return tiles matching an original image stem, including offsets/sizes."""
    tiles: List[Dict[str, int]] = []
    for img in tiled_dataset.images:
        stem = Path(img.file_name).stem
        if not stem.startswith(f"{base_name}_tile_"):
            continue
        match = _TILE_PATTERN.search(img.file_name)
        x_offset = int(match.group(1)) if match else 0
        y_offset = int(match.group(2)) if match else 0
        tiles.append(
            {
                "file": img.file_name,
                "x": x_offset,
                "y": y_offset,
                "w": int(img.width or 640),
                "h": int(img.height or 640),
                "id": img.id,
            }
        )
    # Order tiles by grid position for consistent coloring/labels
    return sorted(tiles, key=lambda t: (t["y"], t["x"]))


def _load_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _draw_tiles_on_image(
    image: Image.Image,
    tiles: List[Dict[str, int]],
    categories: Dict[int, str],
    annotations: List[CocoAnnotation],
    show_annotations: bool,
) -> Image.Image:
    """Overlay tile boundaries (and optionally annotations) on the original image."""
    vis = BoundingBoxVisualizer()
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Pick a font size proportional to the image for readability
    base_size = max(img.width, img.height)
    font_size = max(16, int(base_size / 80))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    colors = vis.colors
    line_width = max(3, int(base_size / 400))

    for idx, tile in enumerate(tiles):
        color = colors[idx % len(colors)]
        x1, y1 = tile["x"], tile["y"]
        x2, y2 = x1 + tile["w"], y1 + tile["h"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Label: tile index + size
        label = f"T{idx+1} {tile['w']}x{tile['h']}"
        bbox = draw.textbbox((0, 0), label, font=font)
        label_w = bbox[2] - bbox[0]
        label_h = bbox[3] - bbox[1]
        pad = 4
        draw.rectangle(
            [x1, y1, x1 + label_w + pad * 2, y1 + label_h + pad * 2],
            fill=color,
            outline=color,
        )
        draw.text((x1 + pad, y1 + pad), label, fill="white", font=font)

    if show_annotations and annotations:
        img = vis.draw_bounding_boxes(img, annotations, categories, show_labels=True)

    return img


def visualize_slicing(
    original_root: str,
    tiled_root: str,
    mode: str,
    fold: int,
    split: str,
    output_dir: str,
    samples: int,
    target_image: Optional[str],
    show_annotations: bool,
) -> None:
    mode = mode.lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of {VALID_MODES}.")

    original_annotations = _resolve_original_annotation_path(original_root)
    tiled_annotations = _resolve_annotation_path(tiled_root, mode, fold, split)
    tile_dir = _resolve_image_dir(tiled_root, mode, fold, split)

    if not os.path.exists(original_annotations):
        raise FileNotFoundError(f"Original annotations not found: {original_annotations}")
    if not os.path.exists(tiled_annotations):
        raise FileNotFoundError(f"Tiled annotations not found: {tiled_annotations}")
    if not os.path.isdir(tile_dir):
        raise FileNotFoundError(f"Tiled image directory not found: {tile_dir}")

    original_dataset, tiled_dataset = _load_datasets(original_annotations, tiled_annotations)
    categories = {cat.id: cat.name for cat in original_dataset.categories}

    images_to_plot: List[Path] = []
    if target_image:
        match = [img for img in original_dataset.images if img.file_name == target_image]
        if not match:
            raise ValueError(f"Image '{target_image}' not found in original dataset.")
        images_to_plot = match
    else:
        images_to_plot = original_dataset.images[:samples]

    os.makedirs(output_dir, exist_ok=True)
    print(f"Visualizando fronteiras de slicing ({mode.upper()}) para {len(images_to_plot)} imagens...")
    print(f"Tamanho esperado de tile: 640x640 (bordas podem ser menores).")
    print(f"Saída em: {output_dir}")

    for img_info in images_to_plot:
        base_name = Path(img_info.file_name).stem
        tiles = _collect_tiles_for_image(base_name, tiled_dataset)
        if not tiles:
            print(f" - {img_info.file_name}: nenhum tile encontrado (pulando).")
            continue

        original_path = _resolve_original_image_path(original_root, img_info.file_name)
        if not os.path.exists(original_path):
            print(f" - {img_info.file_name}: arquivo não encontrado em {original_path} (pulando).")
            continue

        original_img = _load_image(original_path)
        anns = [ann for ann in original_dataset.annotations if ann.image_id == img_info.id]
        vis_image = _draw_tiles_on_image(
            original_img,
            tiles,
            tile_dir,
            categories,
            anns,
            show_annotations=show_annotations,
        )

        out_file = Path(output_dir) / f"{base_name}_tiling_overlay.jpg"
        vis_image.save(out_file)
        print(f" - {img_info.file_name}: {len(tiles)} tiles -> {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize fronteiras e recortes gerados pelo slicing SAHI/ASAHI."
    )
    parser.add_argument("--original", default="./dataset", help="Pasta do dataset original (default: ./dataset)")
    parser.add_argument("--tiled", default="./output", help="Pasta do dataset fatiado (default: ./output)")
    parser.add_argument(
        "--mode",
        choices=sorted(VALID_MODES),
        default="asahi",
        help="Modo de slicing utilizado (asahi/sahi). Default: asahi",
    )
    parser.add_argument("--fold", type=int, default=1, help="Fold a ler no modo ASAHI (default: 1)")
    parser.add_argument("--split", default="train", help="Split do dataset (train/val/test). Default: train")
    parser.add_argument("--output", default="./sahi_vis_640", help="Pasta de saída das visualizações.")
    parser.add_argument("--samples", type=int, default=5, help="Número de imagens para visualizar (default: 5)")
    parser.add_argument("--image", help="Nome exato do arquivo de imagem para visualizar (prioritário).")
    parser.add_argument(
        "--with-annotations",
        action="store_true",
        help="Desenhar também as bounding boxes originais sobre a imagem.",
    )

    args = parser.parse_args()

    visualize_slicing(
        original_root=args.original,
        tiled_root=args.tiled,
        mode=args.mode,
        fold=args.fold,
        split=args.split,
        output_dir=args.output,
        samples=args.samples,
        target_image=args.image,
        show_annotations=args.with_annotations,
    )


if __name__ == "__main__":
    main()
