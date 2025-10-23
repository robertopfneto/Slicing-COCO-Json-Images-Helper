from __future__ import annotations

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
        longest_side = max(image_width, image_height)
        ls_threshold = self.config.ls_threshold
        layout_mode = "compact" if longest_side <= ls_threshold else "expanded"

        if image_width >= image_height:
            cols = 3 if layout_mode == "compact" else 4
            rows = max(1, (6 if layout_mode == "compact" else 12) // cols)
        else:
            rows = 3 if layout_mode == "compact" else 4
            cols = max(1, (6 if layout_mode == "compact" else 12) // rows)

        tile_length = max(1, min(self.config.restrict_size, max(image_width, image_height)))
        overlap_px = int(round(tile_length * self.config.overlap_ratio))
        overlap_px = max(0, min(overlap_px, tile_length - 1))
        stride = max(tile_length - overlap_px, 1)

        x_positions = self._build_axis_positions(image_width, tile_length, stride, cols)
        y_positions = self._build_axis_positions(image_height, tile_length, stride, rows)

        bboxes: List[Tuple[int, int, int, int]] = []
        for y in y_positions:
            for x in x_positions:
                right = min(image_width, x + tile_length)
                bottom = min(image_height, y + tile_length)
                bbox = (int(x), int(y), int(right), int(bottom))
                if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                    continue
                bboxes.append(bbox)

        baseline_bboxes = self._compute_slice_bboxes(image_width, image_height)
        baseline_total = len(baseline_bboxes) if baseline_bboxes else len(bboxes)
        actual_total = len(bboxes)
        ratio = actual_total / baseline_total if baseline_total else 1.0
        redundancy_reduction = max(0.0, 1.0 - ratio)
        time_reduction = redundancy_reduction  # proxy metric - assumes linear relation

        summary = {
            "mode": "ASAHI",
            "layout": layout_mode,
            "cols": len(x_positions),
            "rows": len(y_positions),
            "tile_length": tile_length,
            "overlap_px": overlap_px,
            "overlap_ratio": self.config.overlap_ratio,
            "stride": stride,
            "target_total": cols * rows,
            "actual_total": actual_total,
            "baseline_total": baseline_total,
            "redundancy_reduction": redundancy_reduction,
            "time_reduction": time_reduction,
            "longest_side": longest_side,
            "ls_threshold": ls_threshold,
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
        print(
            "  [ASAHI] plan "
            f"{summary.get('rows', 0)}x{summary.get('cols', 0)} "
            f"tiles={summary.get('actual_total', 0)} "
            f"tile={summary.get('tile_length', 0)}px "
            f"overlap={summary.get('overlap_ratio', 0.0):.2f} "
            f"redundancy={summary.get('redundancy_reduction', 0.0)*100:.1f}% "
            f"time_savings~{summary.get('time_reduction', 0.0)*100:.1f}%"
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
