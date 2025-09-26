from dataclasses import dataclass
from typing import Tuple, Optional
import os


@dataclass
class TilingConfig:
    tile_size: Tuple[int, int] = (512, 512)
    overlap: int = 0
    min_object_coverage: float = 0.3
    output_format: str = "COCO"
    resize_output: Optional[Tuple[int, int]] = None  # If set, resize tiles to this size after tiling


@dataclass
class DatasetConfig:
    input_path: str = "./dataset/all/train"
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
                    int(os.getenv("TILE_WIDTH", 512)),
                    int(os.getenv("TILE_HEIGHT", 512))
                ),
                overlap=int(os.getenv("TILE_OVERLAP", 0)),
                min_object_coverage=float(os.getenv("MIN_OBJECT_COVERAGE", 0.3)),
                output_format=os.getenv("OUTPUT_FORMAT", "COCO"),
                resize_output=(
                    (int(os.getenv("RESIZE_WIDTH")), int(os.getenv("RESIZE_HEIGHT")))
                    if os.getenv("RESIZE_WIDTH") and os.getenv("RESIZE_HEIGHT")
                    else None
                )
            ),
            dataset=DatasetConfig(
                input_path=os.getenv("INPUT_PATH", "./dataset/all/train"),
                output_path=os.getenv("OUTPUT_PATH", "./output"),
                train_split=float(os.getenv("TRAIN_SPLIT", 0.8)),
                val_split=float(os.getenv("VAL_SPLIT", 0.1)),
                test_split=float(os.getenv("TEST_SPLIT", 0.1))
            ),
            processing=ProcessingConfig(
                batch_size=int(os.getenv("BATCH_SIZE", 32)),
                num_workers=int(os.getenv("NUM_WORKERS", 4)),
                save_original_annotations=bool(os.getenv("SAVE_ORIGINAL", True)),
                generate_tiles_only=bool(os.getenv("TILES_ONLY", False))
            )
        )
