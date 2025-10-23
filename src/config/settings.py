from dataclasses import dataclass
from typing import Tuple, Optional
import os


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
    ignore_negative_samples: bool = False
    verbose: bool = False
    exif_fix: bool = True


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
                auto_slice_resolution=os.getenv("AUTO_SLICE_RESOLUTION", "false").lower()
                in {"1", "true", "yes"},
                min_object_coverage=float(os.getenv("MIN_OBJECT_COVERAGE", 0.3)),
                min_area_ratio=float(os.getenv("MIN_AREA_RATIO", 0.1)),
                output_format=os.getenv("OUTPUT_FORMAT", "COCO"),
                resize_output=(
                    (int(os.getenv("RESIZE_WIDTH")), int(os.getenv("RESIZE_HEIGHT")))
                    if os.getenv("RESIZE_WIDTH") and os.getenv("RESIZE_HEIGHT")
                    else None
                ),
                ignore_negative_samples=os.getenv("IGNORE_NEGATIVE_SAMPLES", "false").lower()
                in {"1", "true", "yes"},
                verbose=os.getenv("SAHI_VERBOSE", "false").lower() in {"1", "true", "yes"},
                exif_fix=os.getenv("SAHI_EXIF_FIX", "true").lower() not in {"0", "false", "no"}
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
                save_original_annotations=bool(os.getenv("SAVE_ORIGINAL", True)),
                generate_tiles_only=bool(os.getenv("TILES_ONLY", False)),
                num_folds=int(os.getenv("NUM_FOLDS", 5)),
                shuffle_folds=os.getenv("SHUFFLE_FOLDS", "true").lower() in {"1", "true", "yes"},
                fold_seed=int(os.getenv("FOLD_SEED", 42))
            )
        )
