#!/usr/bin/env python3
"""
Quick verification script to reconstruct a full image from its saved tiles.

It scans the tiled output (default: ./output/tile/fold_1/train), picks a random
stem, and reassembles all tiles that share that stem using the offsets encoded
in the filename: <stem>_tile_<x>_<y>.jpg.
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def load_plan(output_root: Path) -> Tuple[int, int]:
    plan_path = output_root / "tiling_plan.json"
    if not plan_path.exists():
        return 0, 0
    try:
        with plan_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        target = data.get("target_reconstruction_size") or [0, 0]
        if isinstance(target, (list, tuple)) and len(target) == 2:
            w, h = int(target[0]), int(target[1])
            return max(w, 0), max(h, 0)
    except (OSError, json.JSONDecodeError):
        pass
    return 0, 0


def find_tiles(root: Path) -> Dict[str, List[Path]]:
    tiles: Dict[str, List[Path]] = {}
    pattern = re.compile(r"^(?P<stem>.+)_tile_(?P<x>\d+)_(?P<y>\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
    for path in root.iterdir():
        match = pattern.match(path.name)
        if not match:
            continue
        stem = match.group("stem")
        tiles.setdefault(stem, []).append(path)
    return tiles


def reconstruct(stem: str, tile_paths: List[Path], target_size: Tuple[int, int]) -> Image.Image:
    offsets: List[Tuple[int, int, Path]] = []
    max_w, max_h = target_size
    pattern = re.compile(r"_tile_(?P<x>\d+)_(?P<y>\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

    for path in tile_paths:
        match = pattern.search(path.name)
        if not match:
            continue
        x = int(match.group("x"))
        y = int(match.group("y"))
        offsets.append((x, y, path))

    tiles_loaded: List[Tuple[int, int, Image.Image]] = []
    for x, y, path in offsets:
        img = Image.open(path).convert("RGB")
        tiles_loaded.append((x, y, img))
        max_w = max(max_w, x + img.width)
        max_h = max(max_h, y + img.height)

    canvas = Image.new("RGB", (max_w, max_h), (0, 0, 0))
    for x, y, img in tiles_loaded:
        canvas.paste(img, (x, y))
        img.close()

    return canvas


def main() -> None:
    output_root = Path(os.getenv("OUTPUT_PATH", "./output"))
    fold_dir = output_root / "tile" / "fold_1" / "train"

    if not fold_dir.is_dir():
        raise SystemExit(f"Fold directory not found: {fold_dir}")

    tiles_by_stem = find_tiles(fold_dir)
    if not tiles_by_stem:
        raise SystemExit(f"No tiles found in {fold_dir}")

    stems = sorted(tiles_by_stem.keys())
    stem = random.choice(stems)
    target_size = load_plan(output_root)

    print(f"Reconstructing stem '{stem}' from {len(tiles_by_stem[stem])} tiles...")
    if target_size[0] and target_size[1]:
        print(f"Target size from tiling_plan.json: {target_size[0]}x{target_size[1]} px")

    reconstructed = reconstruct(stem, tiles_by_stem[stem], target_size)
    save_path = output_root / f"reconstructed_{stem}.jpg"
    reconstructed.save(save_path, "JPEG", quality=95)
    print(f"Saved reconstruction to: {save_path}")
    print(f"Canvas size: {reconstructed.width}x{reconstructed.height} px")


if __name__ == "__main__":
    main()
