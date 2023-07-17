from pydantic import BaseModel
from enum import Enum

class DogImage(BaseModel):
    image_bytes: bytes
    task: str
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    top_n: int = 3
