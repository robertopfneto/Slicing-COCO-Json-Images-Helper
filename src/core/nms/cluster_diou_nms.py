from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union


BBox = Tuple[float, float, float, float]


@dataclass
class DetectionRecord:
    """Normalized detection structure used by Cluster-DIoU-NMS."""

    bbox: BBox
    score: float
    category_id: int
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Union["DetectionRecord", Dict[str, Any]]) -> "DetectionRecord":
        if isinstance(payload, DetectionRecord):
            return payload
        bbox = tuple(payload.get("bbox", (0.0, 0.0, 0.0, 0.0)))  # type: ignore[arg-type]
        if len(bbox) != 4:
            raise ValueError("Detection payload must contain a 'bbox' with four values.")
        return cls(
            bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            score=float(payload.get("score", 0.0)),
            category_id=int(payload.get("category_id", -1)),
            extra={
                key: value
                for key, value in payload.items()
                if key not in {"bbox", "score", "category_id"}
            },
        )


def cluster_diou_nms(
    detections: Sequence[Union[DetectionRecord, Dict[str, Any]]],
    iou_threshold: float = 0.5,
    diou_threshold: float = 0.7,
    cluster_iou: float = 0.3,
) -> List[DetectionRecord]:
    """
    Apply Cluster-DIoU-NMS to a detection list.

    The routine first groups detections into loose IoU-based clusters, then applies
    a DIoU-aware suppression inside each cluster while preserving high-confidence
    representatives. The output maintains descending score order.
    """

    if not detections:
        return []

    records = [DetectionRecord.from_payload(det) for det in detections]
    ordered_indices = sorted(range(len(records)), key=lambda idx: records[idx].score, reverse=True)

    suppressed = set()
    kept: List[int] = []

    for anchor_idx in ordered_indices:
        if anchor_idx in suppressed:
            continue

        anchor = records[anchor_idx]
        cluster_members = _collect_cluster(anchor_idx, records, ordered_indices, cluster_iou, suppressed)

        survivors = _suppress_within_cluster(
            anchor_idx=anchor_idx,
            cluster=cluster_members,
            records=records,
            iou_threshold=iou_threshold,
            diou_threshold=diou_threshold,
            suppressed=suppressed,
        )
        kept.extend(survivors)

    kept_sorted = sorted(set(kept), key=lambda idx: records[idx].score, reverse=True)
    return [records[idx] for idx in kept_sorted]


def _collect_cluster(
    anchor_idx: int,
    records: Sequence[DetectionRecord],
    ordered_indices: Iterable[int],
    cluster_iou: float,
    suppressed: set[int],
) -> List[int]:
    members = [anchor_idx]
    anchor = records[anchor_idx]

    for idx in ordered_indices:
        if idx == anchor_idx or idx in suppressed:
            continue
        candidate = records[idx]
        if candidate.category_id != anchor.category_id:
            continue
        if _iou(anchor.bbox, candidate.bbox) >= cluster_iou:
            members.append(idx)

    return members


def _suppress_within_cluster(
    anchor_idx: int,
    cluster: List[int],
    records: Sequence[DetectionRecord],
    iou_threshold: float,
    diou_threshold: float,
    suppressed: set[int],
) -> List[int]:
    survivors: List[int] = []

    for idx in cluster:
        if idx in suppressed:
            continue

        candidate = records[idx]
        keep = True
        for survivor_idx in survivors:
            survivor = records[survivor_idx]
            if candidate.category_id != survivor.category_id:
                continue
            iou_value = _iou(candidate.bbox, survivor.bbox)
            diou_value = _diou(candidate.bbox, survivor.bbox)
            if iou_value > iou_threshold or diou_value > diou_threshold:
                keep = False
                break

        if keep:
            survivors.append(idx)
        else:
            suppressed.add(idx)

    for idx in cluster:
        if idx not in survivors:
            suppressed.add(idx)

    return survivors


def _iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = max(area_a + area_b - inter_area, 1e-6)
    return inter_area / union_area


def _diou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    iou_value = _iou(box_a, box_b)
    if iou_value <= 0.0:
        return 0.0

    center_a = ((ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0)
    center_b = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)

    enclosing_x1 = min(ax1, bx1)
    enclosing_y1 = min(ay1, by1)
    enclosing_x2 = max(ax2, bx2)
    enclosing_y2 = max(ay2, by2)
    diagonal = max((enclosing_x2 - enclosing_x1) ** 2 + (enclosing_y2 - enclosing_y1) ** 2, 1e-6)

    center_distance = (center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2
    return iou_value - (center_distance / diagonal)
