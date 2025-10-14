"""Helpers for the SAGE (Stride-Aligned Grid Extraction) tiling strategy."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import median
from typing import Iterable, List, Optional, Sequence, Tuple

from src.models.coco import CocoAnnotation


@dataclass(frozen=True)
class SageGrid:
    """Container for a generated SAGE grid."""

    boxes: List[Tuple[int, int, int, int]]
    stride: Tuple[float, float]
    layout: Tuple[int, int]


def compute_adaptive_overlap(
    annotations: Iterable[CocoAnnotation],
    tile_size: Tuple[int, int],
    *,
    annotation_filter: Optional[Sequence[int]] = None,
) -> float:
    """Estimate the adaptive overlap ratio O* for SAGE.

    Args:
        annotations: Iterable with all available annotations.
        tile_size: (width, height) of the target tile.
        annotation_filter: Optional set/list of image IDs to consider.

    Returns:
        The overlap ratio O* in the range [0, 0.5]. Defaults to 0 when not enough
        annotations are available.
    """

    tile_w, tile_h = tile_size
    tile_edge = float(min(tile_w, tile_h))

    widths: List[float] = []
    heights: List[float] = []

    image_filter = set(annotation_filter) if annotation_filter else None

    for ann in annotations:
        if image_filter and ann.image_id not in image_filter:
            continue
        _, _, w, h = ann.bbox
        if w > 0 and h > 0:
            widths.append(w)
            heights.append(h)

    if not widths or not heights:
        return 0.0

    w_med = float(median(widths))
    h_med = float(median(heights))
    d = sqrt(w_med * h_med)

    if tile_edge <= 0:
        return 0.0

    ratio = 0.5 * (d / tile_edge)
    return max(0.0, min(ratio, 0.5))


def compute_sage_grid(
    image_shape: Tuple[int, int],
    tile_size: Tuple[int, int],
    overlap: float,
) -> SageGrid:
    """Generate the stride-aligned grid definition using SAGE.

    Args:
        image_shape: Image height and width.
        tile_size: Tile width and height.
        overlap: Overlap ratio (O*).

    Returns:
        SageGrid with bounding boxes (x1, y1, x2, y2).
    """

    height, width = image_shape
    tile_w, tile_h = tile_size

    stride_x = tile_w * (1.0 - overlap)
    stride_y = tile_h * (1.0 - overlap)

    nx = int(round((width - tile_w) / stride_x)) + 1 if stride_x > 0 else 1
    ny = int(round((height - tile_h) / stride_y)) + 1 if stride_y > 0 else 1

    nx = max(nx, 1)
    ny = max(ny, 1)

    if nx > 1:
        stride_x = (width - tile_w) / (nx - 1)
    if ny > 1:
        stride_y = (height - tile_h) / (ny - 1)

    boxes: List[Tuple[int, int, int, int]] = []
    for j in range(ny):
        for i in range(nx):
            x1 = int(round(i * stride_x))
            y1 = int(round(j * stride_y))
            x2 = min(x1 + tile_w, width)
            y2 = min(y1 + tile_h, height)
            x1 = max(0, x2 - tile_w)
            y1 = max(0, y2 - tile_h)
            boxes.append((x1, y1, x2, y2))

    return SageGrid(boxes=boxes, stride=(stride_x, stride_y), layout=(nx, ny))


def iter_sage_boxes(
    image_shape: Tuple[int, int],
    tile_size: Tuple[int, int],
    overlap: float,
):
    """Yield SAGE boxes as tuples (x1, y1, x2, y2)."""

    grid = compute_sage_grid(image_shape, tile_size, overlap)
    for box in grid.boxes:
        yield box
