from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json


@dataclass
class CocoImage:
    id: int
    width: int
    height: int
    file_name: str
    license: Optional[int] = None
    flickr_url: Optional[str] = None
    coco_url: Optional[str] = None
    date_captured: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class CocoCategory:
    id: int
    name: str
    supercategory: str


@dataclass
class CocoAnnotation:
    id: int
    image_id: int
    category_id: int
    segmentation: List[List[float]]
    area: float
    bbox: List[float]  # [x, y, width, height]
    iscrowd: int = 0


@dataclass
class CocoInfo:
    year: int
    version: str
    description: str
    contributor: str
    url: str
    date_created: str


@dataclass
class CocoLicense:
    id: int
    name: str
    url: str


@dataclass
class CocoDataset:
    info: CocoInfo
    licenses: List[CocoLicense]
    images: List[CocoImage]
    annotations: List[CocoAnnotation]
    categories: List[CocoCategory]
    
    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Helper function to safely create dataclass instances
        def safe_create_instance(dataclass_type, data_dict):
            # Get the field names that the dataclass expects
            field_names = {field.name for field in dataclass_type.__dataclass_fields__.values()}
            # Filter the data to only include expected fields
            filtered_data = {k: v for k, v in data_dict.items() if k in field_names}
            return dataclass_type(**filtered_data)
        
        return cls(
            info=safe_create_instance(CocoInfo, data['info']) if 'info' in data and data['info'] else None,
            licenses=[safe_create_instance(CocoLicense, license) for license in data.get('licenses', [])],
            images=[safe_create_instance(CocoImage, img) for img in data['images']],
            annotations=[safe_create_instance(CocoAnnotation, ann) for ann in data['annotations']],
            categories=[safe_create_instance(CocoCategory, cat) for cat in data['categories']]
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'info': self.info.__dict__ if self.info else {},
            'licenses': [license.__dict__ for license in self.licenses],
            'images': [img.__dict__ for img in self.images],
            'annotations': [ann.__dict__ for ann in self.annotations],
            'categories': [cat.__dict__ for cat in self.categories]
        }
    
    def save_json(self, output_path: str):
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)