import os
import numpy as np
from PIL import Image
import json
from keras.models import load_model

MODEL_INPUT_SHAPE = (224, 224)

def get_model(model_path):
    model = load_model(model_path)
    return model

def get_clf_model():
    model_path = 'models/InceptionResNetV2.h5'
    return get_model(model_path)

def clf_breed(model, image: Image, top_n: int=3) -> dict:
    image = preprocess_image(image)
    prediction_arr = model.predict(image)[0]

    with open("breeds_dict.json") as f:
        breed_dict = json.load(f)

    top_n_pred_score = np.argsort(prediction_arr)[-top_n:][::-1]
    top_n_pred_breed = {breed_dict[str(i)]: float(prediction_arr[i]) for i in top_n_pred_score}

    return top_n_pred_breed

def preprocess_image(image: Image) -> np.ndarray:
    """
    Preprocess the image to be used for prediction.
    """
    # Resize the image to the model input shape
    image = image.resize(MODEL_INPUT_SHAPE).convert('RGB')
    image = np.array([np.array(image)]) / 255
    return image
