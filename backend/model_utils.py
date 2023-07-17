import base64, io, json, os
import numpy as np
from PIL import Image
from keras.models import load_model
from ultralytics import YOLO
import cv2

MODEL_INPUT_SHAPE = (224, 224)


def get_model(
        model_path: str
        ):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    model = load_model(model_path)
    return model

def get_clf_model():
    model_path = 'model-binaries/InceptionResNetV2.h5'
    model = get_model(model_path)
    return model

def get_det_model():
    model_path = 'model-binaries/best.torchscript'
    model = YOLO(model_path, task="detect")
    return model

def det_breed(
    model,
    image_bytes: bytes,
    conf_thres: float=0.25,
    iou_thres: float=0.45,
) -> dict:
    """
    This function takes in a model and an image-64-bit data and returns bounding boxes and classes.
    """
    image = preprocess_image(image_bytes, task="det")
    # results = model.predict(image, conf_thres=conf_thres, iou_thres=iou_thres)[0].boxes
    results = model.predict(image)[0].boxes

    data = {
        "cls": results.cls.tolist(),
        "xyxyn": results.xyxyn.tolist(),
        "conf": results.conf.tolist(),
    }
    with open("model-binaries/breeds_dict.json") as f:
        breed_dict = json.load(f)

    data["cls"] = [breed_dict[str(int(i))] for i in data["cls"]]


    return data

def clf_breed(
    model,
    image_bytes: bytes,
    top_n: int=3
) -> dict:
    """
    This function takes in a model and an image-64-bit data and returns the top N breeds with probability percentages.
    """
    image = preprocess_image(image_bytes)
    prediction = model.predict(image)[0]

    with open("model-binaries/breeds_dict.json") as f:
        breed_dict = json.load(f)

    top_n_pred_score = np.argsort(prediction)[-top_n:][::-1]
    top_n_pred_breed = {
                breed_dict[str(i)]: float(prediction[i])
                for i in top_n_pred_score
            }
    return top_n_pred_breed

def preprocess_image(image_bytes: bytes, task="clf") -> np.ndarray:
    """
    Preprocess the image to be used for prediction.
    """
    # Resize the image to the model input shape
    image = Image.open(io.BytesIO(base64.decodebytes(image_bytes)))
    if task == "clf":
        image = image.resize(MODEL_INPUT_SHAPE).convert('RGB')
        image = np.array([np.array(image)]) / 255
    elif task == "det":
        w, h = image.size
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = image.resize(new_w, new_h).convert('RGB')
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image
