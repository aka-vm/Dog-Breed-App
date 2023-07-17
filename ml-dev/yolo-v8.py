# %% [markdown]
# Original Code -
#
# [Kaggle Notebook](https://www.kaggle.com/code/vineetmahajan/dog-breed-detection) |
# [GitHub](https://github.com/aka-vm/Hello-CV/blob/master/Stanford%20Dogs/detection/yolo-v8.ipynb)

# %%
from ultralytics import YOLO
import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt

# %%
model_path = "model-binaries/best.torchscript"


# %%
def detect_breed(image):
    results = model.predict(image)

    return results[0]

# %%
image = "German-Shepherd-dog-Alsatian.jpg"
image = np.array(PIL.Image.open(image))
h, w, _ = image.shape
scale = 640 / max(h, w)
new_h, new_w = int(h * scale), int(w * scale)
image = cv2.resize(image, (new_w, new_h))

# %%
model = YOLO(model_path, task="detect")

result = detect_breed(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
result.boxes

# %%
result_image = result.plot(img = image, line_width=3)

plt.imshow(result_image)

# %%
import ultralytics

ultralytics.__version__

# %%



