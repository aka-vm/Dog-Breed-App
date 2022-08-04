import os
import base64, io
import PIL.Image as Image

from enum import Enum
from typing import Union

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .model_utils import get_clf_model, clf_breed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
clf_model = get_clf_model()

class CVisionOppration(str, Enum):
    predict = "predict"
    detect = "detect"

class DogImage(BaseModel):
    image_bytes: bytes
    top_n: int = 3


app = FastAPI()

@app.get("/")
async def root():
    data = {
        "message": "Hello World",
        }
    return data

@app.post("/vision/{operation}")
async def vision(operation: CVisionOppration, dog_image: DogImage):

    if operation == CVisionOppration.predict:
        image_bytes = dog_image.image_bytes
        image = Image.open(io.BytesIO(base64.decodebytes(image_bytes)))
        respon = clf_breed(clf_model, image, dog_image.top_n)
        return JSONResponse(respon)

    if operation == CVisionOppration.detect:
        return {"message": "Yolo"}

