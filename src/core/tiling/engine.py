from __future__ import annotations

import math
from typing import Generator, Iterable, List, Optional, Tuple, Dict, Any, Sequence, Union

from PIL import Image
from shapely.errors import TopologicalError
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid

from src.config.settings import TilingConfig
from src.models.coco import CocoAnnotation


def calc_ratio_and_slice(orientation: str, slide: int = 1, ratio: float = 0.1) -> Tuple[int, int, float, float]:
    """Replicates SAHI ratio presets for automatic slicing."""
    if orientation == "vertical":
        return slide, slide * 2, ratio, ratio
    if orientation == "horizontal":
        return slide * 2, slide, ratio, ratio
    if orientation == "square":
        return slide, slide, ratio, ratio
    raise ValueError(
        f"Invalid orientation '{orientation}'. Expected one of 'vertical', 'horizontal', or 'square'."
    )


def calc_resolution_factor(resolution: int) -> int:
    """Return floor(log2(resolution)) analogue used by SAHI auto slicing."""
    expo = 0
    while (1 << expo) < resolution:
        expo += 1
    return max(expo - 1, 0)


def calc_aspect_ratio_orientation(width: int, height: int) -> str:
    if width < height:
        return "vertical"
    if width > height:
        return "horizontal"
    return "square"


def calc_slice_and_overlap_params(
    resolution: str, height: int, width: int, orientation: str
) -> Tuple[int, int, int, int]:
    split_row: int
    split_col: int
    overlap_height_ratio: float
    overlap_width_ratio: float

    if resolution == "medium":
        split_row, split_col, overlap_height_ratio, overlap_width_ratio = calc_ratio_and_slice(
            orientation, slide=1, ratio=0.8
        )
    elif resolution == "high":
        split_row, split_col, overlap_height_ratio, overlap_width_ratio = calc_ratio_and_slice(
            orientation, slide=2, ratio=0.4
        )
    elif resolution == "ultra-high":
        split_row, split_col, overlap_height_ratio, overlap_width_ratio = calc_ratio_and_slice(
            orientation, slide=4, ratio=0.4
        )
    else:
        split_row = 1
        split_col = 1
        overlap_width_ratio = 1.0
        overlap_height_ratio = 1.0

    slice_height = max(height // split_col, 1)
    slice_width = max(width // split_row, 1)

    x_overlap = int(slice_width * overlap_width_ratio)
    y_overlap = int(slice_height * overlap_height_ratio)
    return x_overlap, y_overlap, slice_width, slice_height


def get_resolution_selector(resolution: str, height: int, width: int) -> Tuple[int, int, int, int]:
    orientation = calc_aspect_ratio_orientation(width=width, height=height)
    return calc_slice_and_overlap_params(
        resolution=resolution, height=height, width=width, orientation=orientation
    )


def get_auto_slice_params(height: int, width: int) -> Tuple[int, int, int, int]:
    """Port of SAHI auto slice parameter selection."""
    resolution = height * width
    factor = calc_resolution_factor(resolution)
    if factor <= 18:
        return get_resolution_selector("low", height=height, width=width)
    if 18 <= factor < 21:
        return get_resolution_selector("medium", height=height, width=width)
    if 21 <= factor < 24:
        return get_resolution_selector("high", height=height, width=width)
    return get_resolution_selector("ultra-high", height=height, width=width)


def get_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
) -> List[Tuple[int, int, int, int]]:
    """Generate slice bounding boxes following SAHI window traversal."""
    slice_bboxes: List[Tuple[int, int, int, int]] = []
    y_min = 0
    y_max = 0

    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)

    while y_max < image_height:
        x_min = 0
        x_max = 0
        y_max = y_min + slice_height

        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append((xmin, ymin, xmax, ymax))
            else:
                slice_bboxes.append((x_min, y_min, x_max, y_max))
            x_min = max(0, x_max - x_overlap)
        y_min = max(0, y_max - y_overlap)
    return slice_bboxes


def _normalize_geometry(geometry: Optional[BaseGeometry]) -> Optional[BaseGeometry]:
    if geometry is None or geometry.is_empty:
        return None
    if isinstance(geometry, GeometryCollection):
        polygons: List[Polygon] = []
        for geom in geometry.geoms:
            normalized = _normalize_geometry(geom)
            if normalized is None:
                continue
            if isinstance(normalized, Polygon):
                polygons.append(normalized)
            elif isinstance(normalized, MultiPolygon):
                polygons.extend(poly for poly in normalized.geoms if not poly.is_empty)
        if not polygons:
            return None
        if len(polygons) == 1:
            return polygons[0]
        return MultiPolygon(polygons)
    if isinstance(geometry, MultiPolygon):
        polygons = [poly for poly in geometry.geoms if not poly.is_empty]
        if not polygons:
            return None
        if len(polygons) == 1:
            return polygons[0]
        return MultiPolygon(polygons)
    return geometry


def _iter_polygons(geometry: BaseGeometry) -> Iterable[Polygon]:
    normalized = _normalize_geometry(geometry)
    if normalized is None:
        return []
    if isinstance(normalized, Polygon):
        return [normalized]
    if isinstance(normalized, MultiPolygon):
        return [poly for poly in normalized.geoms if not poly.is_empty]
    return []


def _geometry_to_bbox(
    geometry: BaseGeometry, offset_x: float, offset_y: float, scale_factor: float
) -> List[float]:
    minx, miny, maxx, maxy = geometry.bounds
    width = maxx - minx
    height = maxy - miny
    return [
        (minx - offset_x) * scale_factor,
        (miny - offset_y) * scale_factor,
        width * scale_factor,
        height * scale_factor,
    ]


def _geometry_to_segmentation(
    geometry: BaseGeometry, offset_x: float, offset_y: float, scale_factor: float
) -> List[List[float]]:
    segments: List[List[float]] = []
    for polygon in _iter_polygons(geometry):
        if polygon.area <= 0:
            continue
        x_coords, y_coords = polygon.exterior.coords.xy
        coords: List[float] = []
        for x_coord, y_coord in zip(x_coords, y_coords):
            adj_x = (x_coord - offset_x) * scale_factor
            adj_y = (y_coord - offset_y) * scale_factor
            coords.extend([adj_x, adj_y])
        if coords[:2] == coords[-2:]:
            coords = coords[:-2]
        if coords:
            segments.append(coords)
    return segments


class TilingEngine:
    """Tile generator with annotation slicing logic adapted from SAHI/ASAHI."""

    def __init__(self, config: TilingConfig):
        self.config = config
        self.dataset_context: Optional[Dict[str, Any]] = None
        self._latest_plan_summary: Dict[str, Any] = {}

    def set_dataset_context(self, context: Optional[Dict[str, Any]]) -> None:
        """Store dataset-level statistics to drive adaptive logic."""
        self.dataset_context = context

    def generate_tiles(
        self, image: Image.Image
    ) -> Generator[Tuple[Image.Image, Tuple[int, int, int, int], float], None, None]:
        """Yield cropped tiles along with their box and scale factor."""
        img_width, img_height = image.size

        if self.config.adaptive_mode:
            slice_bboxes, plan_summary = self._compute_asahi_tile_plan(img_width, img_height)
        else:
            slice_bboxes = self._compute_slice_bboxes(img_width, img_height)
            plan_summary = {
                "mode": "SAHI",
                "actual_total": len(slice_bboxes),
                "baseline_total": len(slice_bboxes),
            }

        self._latest_plan_summary = plan_summary
        if plan_summary.get("mode") == "ASAHI" and self.config.verbose:
            self._log_adaptive_plan(plan_summary)

        for bbox in slice_bboxes:
            tile = image.crop(bbox)
            scale_factor = self._compute_scale_factor(bbox)
            if self.config.resize_output:
                tile = tile.resize(self.config.resize_output, Image.LANCZOS)
            yield tile, bbox, scale_factor

    def transform_annotations(
        self,
        annotations: List[CocoAnnotation],
        slice_bbox: Tuple[int, int, int, int],
        scale_factor: float = 1.0,
    ) -> List[CocoAnnotation]:
        """Slice annotations against a tile bounding box."""
        tile_geometry = box(*slice_bbox)
        offset_x, offset_y = slice_bbox[0], slice_bbox[1]
        min_ratio = self.config.min_area_ratio or self.config.min_object_coverage

        transformed_annotations: List[CocoAnnotation] = []
        for annotation in annotations:
            try:
                original_geometry = self._annotation_to_geometry(annotation)
            except TopologicalError:
                continue

            if original_geometry is None:
                continue

            original_area = original_geometry.area
            if original_area <= 0:
                continue

            try:
                clipped_geometry = original_geometry.intersection(tile_geometry)
            except TopologicalError:
                continue

            clipped_geometry = _normalize_geometry(clipped_geometry)
            if clipped_geometry is None or clipped_geometry.area <= 0:
                continue

            if min_ratio and clipped_geometry.area / original_area < min_ratio:
                continue

            new_annotation = CocoAnnotation(
                id=annotation.id,
                image_id=annotation.image_id,
                category_id=annotation.category_id,
                segmentation=_geometry_to_segmentation(
                    clipped_geometry, offset_x, offset_y, scale_factor
                ),
                area=clipped_geometry.area * (scale_factor**2),
                bbox=_geometry_to_bbox(clipped_geometry, offset_x, offset_y, scale_factor),
                iscrowd=annotation.iscrowd,
            )
            transformed_annotations.append(new_annotation)

        return transformed_annotations

    def get_last_plan_summary(self) -> Dict[str, Any]:
        return dict(self._latest_plan_summary)

    def apply_cluster_diou_nms(
        self, detections: Sequence[Union[Dict[str, Any], Any]]
    ) -> Sequence[Union[Dict[str, Any], Any]]:
        """
        Optionally apply Cluster-DIoU-NMS post-processing to a batch of detections.
        The method preserves the original payload type (dict or DetectionRecord).
        """
        if not self.config.cluster_diou_nms or not detections:
            return detections

        try:
            from src.core.nms import DetectionRecord, cluster_diou_nms
        except ImportError:
            return detections

        records = cluster_diou_nms(detections)  # type: ignore[arg-type]

        sample = detections[0]
        if isinstance(sample, DetectionRecord):
            return records

        normalised: List[Dict[str, Any]] = []
        for record in records:
            payload = dict(record.extra)
            payload.update(
                {
                    "bbox": list(record.bbox),
                    "score": record.score,
                    "category_id": record.category_id,
                }
            )
            normalised.append(payload)
        return normalised

    def _compute_asahi_tile_plan(
        self, image_width: int, image_height: int
    ) -> Tuple[List[Tuple[int, int, int, int]], Dict[str, Any]]:
        """
        Compute the adaptive slicing layout following the ASAHI algorithm
        (Adaptive Slicing-Aided Hyper Inference, Zhang et al., Remote Sens. 2023).

        The method adaptively adjusts the tile size `p` according to image
        resolution and overlap ratio `l`, maintaining a fixed number of tiles
        (6 or 12) depending on the adaptive threshold `LS`.

        Behaviour summary:
          • If max(W, H) ≤ LS → compact mode (approx. 6 tiles baseline)
          • If max(W, H) > LS → expanded mode (approx. 12 tiles baseline)
          • Additional rows/cols are added automatically when the 640px limit
            would otherwise leave uncovered regions.
          • Each tile is cropped with overlap ratio l and later resized to 640×640.
          • Redundancy and time reduction are computed per ASAHI Eq. (8–10).
        """
        W = float(max(image_width, 1))
        H = float(max(image_height, 1))
        requested_overlap_ratio = float(self.config.overlap_ratio or 0.0)
        overlap_ratio = min(max(requested_overlap_ratio, 0.0), 0.95)

        raw_restrict = int(self.config.restrict_size or 0)
        restrict_size = max(raw_restrict, 640) if raw_restrict > 0 else 640
        default_ls = restrict_size * (4 - 3 * overlap_ratio) + 1
        configured_ls = float(self.config.ls_threshold or 0.0)
        ls_threshold = configured_ls if configured_ls > 0 else default_ls

        longest_side = max(W, H)
        layout_mode = "compact" if longest_side <= ls_threshold else "expanded"

        epsilon = 1e-6
        if layout_mode == "compact":
            denom_w = max(3 - 2 * overlap_ratio, epsilon)
            denom_h = max(2 - overlap_ratio, epsilon)
        else:
            denom_w = max(4 - 3 * overlap_ratio, epsilon)
            denom_h = max(3 - 2 * overlap_ratio, epsilon)

        p_w = W / denom_w + 1.0
        p_h = H / denom_h + 1.0
        p = max(p_w, p_h)
        tile_length = max(1, int(math.ceil(p)))
        tile_length = min(tile_length, restrict_size, 640)

        overlap_px = int(round(tile_length * overlap_ratio))
        overlap_px = max(0, min(overlap_px, tile_length - 1))
        stride = max(1, tile_length - overlap_px)

        def _compute_axis_positions(
            axis_size: int, tile_extent: int, step: int, target_min: int
        ) -> List[int]:
            if axis_size <= tile_extent or step <= 0:
                return [0]
            max_offset = max(axis_size - tile_extent, 0)
            positions: List[int] = [0]
            pos = step
            while pos < max_offset:
                positions.append(int(round(pos)))
                pos += step
            positions.append(int(max_offset))
            positions = sorted(set(max(0, min(p, max_offset)) for p in positions))
            if target_min > 1 and len(positions) < target_min and max_offset > 0:
                linear_step = max_offset / (target_min - 1)
                positions = [int(round(linear_step * idx)) for idx in range(target_min)]
            if positions:
                positions[0] = 0
                positions[-1] = int(max_offset)
            return positions or [0]

        max_stride = max(stride, 1)
        cols_estimate = (
            int(math.ceil(max(image_width - tile_length, 0) / max_stride)) + 1
            if image_width > tile_length
            else 1
        )
        rows_estimate = (
            int(math.ceil(max(image_height - tile_length, 0) / max_stride)) + 1
            if image_height > tile_length
            else 1
        )

        x_positions = _compute_axis_positions(
            image_width, tile_length, stride, max(cols_estimate, 1)
        )
        y_positions = _compute_axis_positions(
            image_height, tile_length, stride, max(rows_estimate, 1)
        )

        a = len(x_positions)
        b = len(y_positions)

        bboxes: List[Tuple[int, int, int, int]] = []
        for y in y_positions:
            for x in x_positions:
                right = min(image_width, x + tile_length)
                bottom = min(image_height, y + tile_length)
                if right <= x or bottom <= y:
                    continue
                bboxes.append((int(x), int(y), int(right), int(bottom)))

        if not bboxes:
            bboxes = [(0, 0, image_width, image_height)]

        baseline_bboxes = self._compute_slice_bboxes(image_width, image_height)
        baseline_total = len(baseline_bboxes) if baseline_bboxes else len(bboxes)
        actual_total = len(bboxes)
        tile_ratio = actual_total / baseline_total if baseline_total else 1.0
        tile_count_reduction = max(0.0, 1.0 - tile_ratio)

        unique_x_widths = {
            x: min(tile_length, max(image_width - x, 0))
            for x in x_positions
        }
        unique_y_heights = {
            y: min(tile_length, max(image_height - y, 0))
            for y in y_positions
        }

        redundancy_x = max(
            0.0, float(sum(unique_x_widths.values())) - float(image_width)
        )
        redundancy_y = max(
            0.0, float(sum(unique_y_heights.values())) - float(image_height)
        )
        sredundancy = redundancy_x * H + redundancy_y * W - redundancy_x * redundancy_y
        sredundancy = max(0.0, sredundancy)
        sarea = W * H if W > 0 and H > 0 else 0.0
        redundancy_ratio = (sredundancy / sarea) if sarea > 0 else 0.0
        redundancy_ratio = max(0.0, min(redundancy_ratio, 1.0))
        time_reduction = redundancy_ratio

        actual_overlap_ratio = overlap_px / tile_length if tile_length else 0.0

        summary = {
            "mode": "ASAHI",
            "layout": layout_mode,
            "a": a,
            "b": b,
            "cols": a,
            "rows": b,
            "tile_length": tile_length,
            "tile_size": (tile_length, tile_length),
            "stride": stride,
            "stride_xy": (stride, stride),
            "overlap_px": overlap_px,
            "overlap_px_xy": (overlap_px, overlap_px),
            "overlap_ratio": actual_overlap_ratio,
            "overlap_ratio_x": actual_overlap_ratio,
            "overlap_ratio_y": actual_overlap_ratio,
            "overlap_ratio_requested": requested_overlap_ratio,
            "target_total": a * b,
            "tiles_total": actual_total,
            "actual_total": actual_total,
            "baseline_total": baseline_total,
            "tile_count_reduction": tile_count_reduction,
            "redundancy_reduction": redundancy_ratio,
            "redundancy_ratio": redundancy_ratio,
            "time_reduction": time_reduction,
            "longest_side": longest_side,
            "ls_threshold": ls_threshold,
            "resize_target": self.config.resize_output or (640, 640),
        }

        return bboxes, summary

    def _build_axis_positions(
        self, axis_size: int, tile_length: int, stride: int, target_count: int
    ) -> List[int]:
        if target_count <= 1 or axis_size <= tile_length:
            return [0]

        max_offset = max(axis_size - tile_length, 0)
        positions = [
            int(round(min(idx * stride, max_offset)))
            for idx in range(target_count)
        ]

        if len(set(positions)) < target_count and target_count > 1 and max_offset > 0:
            step = max_offset / (target_count - 1)
            positions = [int(round(step * idx)) for idx in range(target_count)]

        positions[-1] = max_offset
        return positions

    def _log_adaptive_plan(self, summary: Dict[str, Any]) -> None:
        rows = int(summary.get("rows") or summary.get("b") or 0)
        cols = int(summary.get("cols") or summary.get("a") or 0)
        tiles_total = summary.get("tiles_total", summary.get("actual_total", 0))

        tile_w, tile_h = summary.get("tile_size") or (
            summary.get("tile_length", 0),
            summary.get("tile_length", 0),
        )
        stride_x, stride_y = summary.get("stride_xy") or (
            summary.get("stride", 0),
            summary.get("stride", 0),
        )
        overlap_px_x, overlap_px_y = summary.get("overlap_px_xy") or (
            summary.get("overlap_px", 0),
            summary.get("overlap_px", 0),
        )
        overlap_ratio_x = summary.get("overlap_ratio_x", summary.get("overlap_ratio", 0.0))
        overlap_ratio_y = summary.get("overlap_ratio_y", overlap_ratio_x)
        redundancy_pct = summary.get("redundancy_reduction", 0.0) * 100
        time_pct = summary.get("time_reduction", 0.0) * 100
        layout = summary.get("layout", "?")
        ls_threshold = summary.get("ls_threshold", 0)
        longest = summary.get("longest_side", 0)
        print(
            "  [ASAHI] plan "
            f"{rows}x{cols} tiles={tiles_total} "
            f"tile={int(tile_w)}x{int(tile_h)}px "
            f"stride={int(stride_x)}x{int(stride_y)}px "
            f"overlap={overlap_ratio_x:.2f}/{overlap_ratio_y:.2f} "
            f"({int(overlap_px_x)}px/{int(overlap_px_y)}px) "
            f"redundancy={redundancy_pct:.1f}% time_savings~{time_pct:.1f}% "
            f"| layout={layout} LS={int(round(ls_threshold))} longest={int(round(longest))}"
        )

    def _compute_slice_bboxes(self, image_width: int, image_height: int) -> List[Tuple[int, int, int, int]]:
        if self.config.auto_slice_resolution:
            x_overlap, y_overlap, slice_width, slice_height = get_auto_slice_params(
                height=image_height, width=image_width
            )
        else:
            slice_width, slice_height = self.config.tile_size
            if slice_width <= 0 or slice_height <= 0:
                raise ValueError("Tile size must contain positive values.")
            overlap_width_ratio = self._resolve_overlap_ratio(
                self.config.overlap_width_ratio, self.config.overlap, slice_width
            )
            overlap_height_ratio = self._resolve_overlap_ratio(
                self.config.overlap_height_ratio, self.config.overlap, slice_height
            )
            x_overlap = int(overlap_width_ratio * slice_width)
            y_overlap = int(overlap_height_ratio * slice_height)

        overlap_width_ratio = x_overlap / slice_width if slice_width else 0.0
        overlap_height_ratio = y_overlap / slice_height if slice_height else 0.0

        overlap_width_ratio = max(0.0, min(overlap_width_ratio, 1.0))
        overlap_height_ratio = max(0.0, min(overlap_height_ratio, 1.0))

        return get_slice_bboxes(
            image_height=image_height,
            image_width=image_width,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )

    @staticmethod
    def _resolve_overlap_ratio(
        configured_ratio: Optional[float], pixel_overlap: int, tile_extent: int
    ) -> float:
        if configured_ratio is not None:
            return max(0.0, min(configured_ratio, 1.0))
        if tile_extent <= 0 or pixel_overlap <= 0:
            return 0.0
        return max(0.0, min(pixel_overlap / tile_extent, 1.0))

    def _compute_scale_factor(self, bbox: Tuple[int, int, int, int]) -> float:
        if not self.config.resize_output:
            return 1.0

        tile_width = bbox[2] - bbox[0]
        tile_height = bbox[3] - bbox[1]
        if tile_width == 0 or tile_height == 0:
            return 1.0

        resize_width, resize_height = self.config.resize_output
        if resize_width <= 0 or resize_height <= 0:
            return 1.0

        return min(resize_width / tile_width, resize_height / tile_height)

    def _annotation_to_geometry(self, annotation: CocoAnnotation) -> Optional[BaseGeometry]:
        geometry: Optional[BaseGeometry] = None

        if annotation.segmentation:
            polygons: List[Polygon] = []
            for segment in annotation.segmentation:
                if len(segment) < 6:
                    continue
                coords = list(zip(segment[0::2], segment[1::2]))
                polygon = Polygon(coords)
                if not polygon.is_valid:
                    polygon = make_valid(polygon)
                normalized = _normalize_geometry(polygon)
                if normalized is None:
                    continue
                if isinstance(normalized, Polygon):
                    polygons.append(normalized)
                elif isinstance(normalized, MultiPolygon):
                    polygons.extend(poly for poly in normalized.geoms if not poly.is_empty)
            if polygons:
                geometry = MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]

        if geometry is None:
            x, y, width, height = annotation.bbox
            geometry = box(x, y, x + width, y + height)

        if geometry.is_empty:
            return None

        geometry = make_valid(geometry)
        return _normalize_geometry(geometry)
