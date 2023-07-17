from typing import Union

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from models import DogImage
from model_utils import get_clf_model, clf_breed, get_det_model, det_breed


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="backend/templates")

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/predict-breed/")
def predict_breed(request: Request):
    return templates.TemplateResponse("result.html", {"request": request})

@app.post("/api/predict-breed/")
async def predict_breed(dog_image: DogImage):
    image_bytes = dog_image.image_bytes
    task = dog_image.task
    if task == "clf":
        clf_model = get_clf_model()
        respon = clf_breed(clf_model, image_bytes, dog_image.top_n)

    elif task == "det":
        det_model = get_det_model()
        respon = det_breed(det_model, image_bytes, dog_image.conf_thres, dog_image.iou_thres)

    return JSONResponse(respon)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0", port=8000)