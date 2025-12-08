from dataclasses import dataclass
from typing import Tuple, Optional
import os


_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    return default


@dataclass
class TilingConfig:
    tile_size: Tuple[int, int] = (640, 640)
    overlap: int = 40
    overlap_height_ratio: Optional[float] = None
    overlap_width_ratio: Optional[float] = None
    auto_slice_resolution: bool = False
    min_object_coverage: float = 0.3
    min_area_ratio: float = 0.1
    output_format: str = "COCO"
    resize_output: Optional[Tuple[int, int]] = None  # If set, resize tiles to this size after tiling
    ignore_negative_samples: bool = True
    verbose: bool = False
    exif_fix: bool = True
    adaptive_mode: bool = True
    auto_overlap: bool = True
    restrict_size: int = 640
    overlap_ratio: float = 0.15
    ls_threshold: float = 0.0
    cluster_diou_nms: bool = False


@dataclass
class DatasetConfig:
    input_path: str = "./dataset"
    output_path: str = "./output"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class ProcessingConfig:
    batch_size: int = 32
    num_workers: int = 4
    save_original_annotations: bool = True
    generate_tiles_only: bool = False
    num_folds: int = 5
    shuffle_folds: bool = True
    fold_seed: int = 42


@dataclass
class AppConfig:
    tiling: TilingConfig
    dataset: DatasetConfig
    processing: ProcessingConfig
    
    @classmethod
    def from_env(cls):
        restrict_size = int(os.getenv("ASAHI_RESTRICT_SIZE", 640))
        overlap_ratio = float(os.getenv("ASAHI_OVERLAP_RATIO", 0.15))
        ls_threshold_env = os.getenv("ASAHI_LS_THRESHOLD")
        try:
            parsed_ls = float(ls_threshold_env) if ls_threshold_env is not None else None
        except ValueError:
            parsed_ls = None
        calculated_ls = restrict_size * (4 - 3 * overlap_ratio) + 1
        ls_threshold = parsed_ls if parsed_ls is not None else calculated_ls

        return cls(
            tiling=TilingConfig(
                tile_size=(
                    int(os.getenv("TILE_WIDTH", 640)),
                    int(os.getenv("TILE_HEIGHT", 640))
                ),
                overlap=int(os.getenv("TILE_OVERLAP", 40)),
                overlap_height_ratio=(
                    float(os.getenv("OVERLAP_HEIGHT_RATIO"))
                    if os.getenv("OVERLAP_HEIGHT_RATIO") is not None
                    else None
                ),
                overlap_width_ratio=(
                    float(os.getenv("OVERLAP_WIDTH_RATIO"))
                    if os.getenv("OVERLAP_WIDTH_RATIO") is not None
                    else None
                ),
                auto_slice_resolution=_env_bool("AUTO_SLICE_RESOLUTION", False),
                min_object_coverage=float(os.getenv("MIN_OBJECT_COVERAGE", 0.3)),
                min_area_ratio=float(os.getenv("MIN_AREA_RATIO", 0.1)),
                output_format=os.getenv("OUTPUT_FORMAT", "COCO"),
                resize_output=(
                    (int(os.getenv("RESIZE_WIDTH")), int(os.getenv("RESIZE_HEIGHT")))
                    if os.getenv("RESIZE_WIDTH") and os.getenv("RESIZE_HEIGHT")
                    else None
                ),
                ignore_negative_samples=_env_bool("IGNORE_NEGATIVE_SAMPLES", True),
                verbose=_env_bool("SAHI_VERBOSE", False),
                exif_fix=_env_bool("SAHI_EXIF_FIX", True),
                adaptive_mode=_env_bool("ASAHI_ADAPTIVE_MODE", True),
                auto_overlap=_env_bool("ASAHI_AUTO_OVERLAP", True),
                restrict_size=restrict_size,
                overlap_ratio=overlap_ratio,
                ls_threshold=ls_threshold,
                cluster_diou_nms=_env_bool("ASAHI_CLUSTER_DIOU_NMS", False),
            ),
            dataset=DatasetConfig(
                input_path=os.getenv("INPUT_PATH", "./dataset"),
                output_path=os.getenv("OUTPUT_PATH", "./output"),
                train_split=float(os.getenv("TRAIN_SPLIT", 0.8)),
                val_split=float(os.getenv("VAL_SPLIT", 0.1)),
                test_split=float(os.getenv("TEST_SPLIT", 0.1))
            ),
            processing=ProcessingConfig(
                batch_size=int(os.getenv("BATCH_SIZE", 32)),
                num_workers=int(os.getenv("NUM_WORKERS", 4)),
                save_original_annotations=_env_bool("SAVE_ORIGINAL", True),
                generate_tiles_only=_env_bool("TILES_ONLY", False),
                num_folds=int(os.getenv("NUM_FOLDS", 5)),
                shuffle_folds=_env_bool("SHUFFLE_FOLDS", True),
                fold_seed=int(os.getenv("FOLD_SEED", 42))
            )
        )
