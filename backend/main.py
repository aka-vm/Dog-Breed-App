from typing import Union

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from .models import DogImage
from .model_utils import get_clf_model, clf_breed


clf_model = get_clf_model()

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World",}

@app.post("/predict-breed/")
async def predict_breed(dog_image: DogImage):
    image_bytes = dog_image.image_bytes
    respon = clf_breed(clf_model, image_bytes, dog_image.top_n)
    return JSONResponse(respon)
