#!/usr/bin/env python3
"""Calcula a area total (px^2) de regioes com overlap para cada imagem do dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from src.models.coco import CocoDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula a area total sobreposta (px^2) gerada pelo tiling em cada imagem."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./dataset",
        help="Pasta raiz do dataset Roboflow (contendo subpastas train/val/test).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Subpasta a ser analisada (train, val ou test).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("LARGURA", "ALTURA"),
        help="Dimensao do tile em pixels (largura altura).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Overlap entre tiles em pixels (stride = tile_size - overlap).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="(Opcional) Caminho para salvar o relatorio em JSON.",
    )
    return parser.parse_args()


def generate_tile_offsets(
    img_width: int, img_height: int, tile_width: int, tile_height: int, overlap: int
) -> List[Tuple[int, int]]:
    """Replica a logica do TilingEngine para obter os offsets dos tiles sem carregar as imagens."""
    step_x = tile_width - overlap
    step_y = tile_height - overlap

    if step_x <= 0 or step_y <= 0:
        raise ValueError("Overlap precisa ser menor que as dimensoes do tile para manter stride positivo.")

    positions: List[Tuple[int, int]] = []

    # Grid principal
    for y in range(0, img_height - tile_height + 1, step_y):
        for x in range(0, img_width - tile_width + 1, step_x):
            positions.append((x, y))

    # Bordas direita e inferior (tiles ancorados)
    if img_width % step_x != 0:
        x = img_width - tile_width
        for y in range(0, img_height - tile_height + 1, step_y):
            positions.append((x, y))

    if img_height % step_y != 0:
        y = img_height - tile_height
        for x in range(0, img_width - tile_width + 1, step_x):
            positions.append((x, y))

    # Canto inferior direito
    if img_width % step_x != 0 and img_height % step_y != 0:
        positions.append((img_width - tile_width, img_height - tile_height))

    return positions


def covered_height(intervals: List[Tuple[int, int]], min_cover: int = 2) -> int:
    """Calcula o comprimento em Y coberto por pelo menos `min_cover` intervalos."""
    events = []
    for y1, y2 in intervals:
        events.append((y1, 1))
        events.append((y2, -1))

    events.sort()
    height = 0
    active = 0

    for idx in range(len(events) - 1):
        y, delta = events[idx]
        active += delta
        next_y = events[idx + 1][0]
        if active >= min_cover and next_y > y:
            height += next_y - y

    return height


def overlap_area(rectangles: List[Tuple[int, int, int, int]]) -> int:
    """Area (px^2) coberta por dois ou mais retangulos."""
    events = []
    for x1, y1, x2, y2 in rectangles:
        events.append((x1, 1, y1, y2))
        events.append((x2, -1, y1, y2))

    events.sort(key=lambda e: e[0])

    active: List[Tuple[int, int]] = []
    prev_x = None
    overlap = 0

    for x, delta, y1, y2 in events:
        if prev_x is not None and x > prev_x and active:
            width = x - prev_x
            height = covered_height(active, min_cover=2)
            overlap += width * height

        if delta == 1:
            active.append((y1, y2))
        else:
            try:
                active.remove((y1, y2))
            except ValueError:
                pass  # Seguranca defensiva para casos raros

        prev_x = x

    return overlap


def calculate_image_overlap(
    img_width: int, img_height: int, tile_size: Tuple[int, int], overlap_px: int
) -> Tuple[int, int]:
    """Retorna (overlap_area_px2, tile_count) para uma imagem."""
    tile_w, tile_h = tile_size
    positions = generate_tile_offsets(img_width, img_height, tile_w, tile_h, overlap_px)

    if not positions:
        return 0, 0

    rectangles = [(x, y, x + tile_w, y + tile_h) for x, y in positions]
    overlap_px2 = overlap_area(rectangles)
    return overlap_px2, len(positions)


def main() -> None:
    args = parse_args()

    annotations_path = Path(args.dataset) / args.split / "_annotations.coco.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Arquivo de anotacoes nao encontrado: {annotations_path}")

    dataset = CocoDataset.from_json(str(annotations_path))
    tile_size = (args.tile_size[0], args.tile_size[1])

    print(f"Dataset: {annotations_path} | Imagens: {len(dataset.images)}")
    print(f"Tile: {tile_size[0]}x{tile_size[1]} px | Overlap: {args.overlap} px")

    results = []
    total_overlap = 0

    for img in dataset.images:
        overlap_px2, tile_count = calculate_image_overlap(img.width, img.height, tile_size, args.overlap)
        image_area = img.width * img.height
        overlap_ratio = (overlap_px2 / image_area) if image_area else 0
        overlap_percent = overlap_ratio * 100

        results.append(
            {
                "image_id": img.id,
                "file_name": img.file_name,
                "width": img.width,
                "height": img.height,
                "tiles": tile_count,
                "overlap_area_px2": overlap_px2,
                "overlap_ratio": round(overlap_ratio, 6),
                "overlap_percent": round(overlap_percent, 4),
            }
        )

        total_overlap += overlap_px2
        print(
            f"- {img.file_name}: overlap={overlap_px2} px^2 | tiles={tile_count} | "
            f"overlap/img={overlap_ratio:.4f} ({overlap_percent:.2f}%)"
        )

    summary = {
        "config": {"tile_size": list(tile_size), "overlap": args.overlap, "split": args.split},
        "images_analyzed": len(results),
        "total_overlap_px2": total_overlap,
        "avg_overlap_ratio": round(
            sum(item["overlap_ratio"] for item in results) / len(results), 6
        ) if results else 0,
        "avg_overlap_percent": round(
            sum(item["overlap_percent"] for item in results) / len(results), 4
        ) if results else 0,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"summary": summary, "images": results}, f, indent=2)
        print(f"\nRelatorio salvo em: {output_path}")
    else:
        print("\nNenhum caminho de saida informado; relatorio nao foi salvo.")


if __name__ == "__main__":
    main()
