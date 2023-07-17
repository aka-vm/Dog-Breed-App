# %% [markdown]
# # InceptionResNetV2
#
# This Notebook trains an InceptionResNetV2 model with the Stanford Dogs dataset, pretrained on ImageNet.

# %%
import os
import pathlib
import sys
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

# %%
import paths

REPO_DIR = paths.get_repo_path()
ROOT_DIR = REPO_DIR / "ml-dev"
DATA_BASE_PATH = paths.get_data_path() / "stanford-dogs-dataset"

RANDOM_SEED = 42

os.chdir(REPO_DIR)

# %% [markdown]
# ### Data Loading

# %%
dogs_df_path = DATA_BASE_PATH / "dogs_df.csv"

dogs_df = pd.read_csv(dogs_df_path)
print(dogs_df.shape[0])
dogs_df.head()

# %%
with open(ROOT_DIR / "breeds_dict.json", "r") as f:
    breed_dict = json.load(f)
len(breed_dict)

# %%
RANDOM_SEED = 42
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

VALIDATION_SPLIT = 0.2

BATCH_SIZE = 32

CLASS_NAMES = list(breed_dict.values())
NUM_CLASSES = len(CLASS_NAMES)

# MODEL
MODEL_PATH = REPO_DIR / "model-binaries"
LOG_PATH = ROOT_DIR / "log"

# %%
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# %%
from sklearn.model_selection import train_test_split

train_data_df, val_data_df = train_test_split(
                                    dogs_df,
                                    test_size=VALIDATION_SPLIT,
                                    stratify=dogs_df["breed"],
                                    )

# %%
train_data_df_dublicated = pd.concat([train_data_df for _ in range(2)]).sample(frac=1)

# %% [markdown]
# ### Data Augmentation

# %%
train_generator = ImageDataGenerator(
    rescale=1./255,

    horizontal_flip=True,
    # vertical_flip=True,
    rotation_range=36,

    height_shift_range=0.1,       # No need to shift the image
    width_shift_range=0.1,
    zoom_range=0.15,

    shear_range=0.1,              # Seems to be useful
    brightness_range = [0.75, 1.25],
)

val_generator = ImageDataGenerator(
    rescale=1./255,
    )

# %%
train_images = train_generator.flow_from_dataframe(
    train_data_df_dublicated,
    x_col="image_path",
    y_col="breed",

    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_images = val_generator.flow_from_dataframe(
    val_data_df,
    x_col="image_path",
    y_col="breed",

    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,

    shuffle=False,
)

test_images = val_images

# %%
num_rows = 5
num_cols = 5

plt.figure(figsize=(20, 15))

images, labels = train_images.next()
for i in range(num_cols * num_rows):
    plt.subplot(num_cols, num_rows, i + 1)
    plt.imshow(images[i])
    plt.title(CLASS_NAMES[labels[i].argmax()])
    plt.axis('off')

plt.show()

# %% [markdown]
# ### Model

# %%
INPUT_SHAPE = train_images.next()[0][0].shape
TRAIN_MODELS = True
TRAIN_MODELS = False
LEARNING_RATE = 20e-5

# %%
from tensorflow.keras.metrics import top_k_categorical_accuracy
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, log_loss

def predict_label(images, model):
    predictions = model.predict(images)
    return predictions.argmax(axis=1)


# ploting the model training history
def plot_model_performance(history, figsize=(10, 10)):
    preformance = {key: val for key, val in history.history.items() if "loss" not in key}
    losses = {key: val for key, val in history.history.items() if "loss" in key}

    plt.figure(figsize=figsize)
    plt.title('Model Performance')
    for key, val in preformance.items():
        plt.plot(val, label=key)
    plt.legend(preformance.keys())
    plt.xlabel('Epoch')

    plt.figure(figsize=figsize)
    plt.title('Model Losses')
    for key, val in losses.items():
        plt.plot(val, label=key)
    plt.legend(losses.keys())
    plt.xlabel('Epoch')

    plt.show()

def compute_performance_metrics(y, y_pred, verbose=1):
    # labels = test_images_.y.argmax(axis=1)
    labels = y
    labels_cat = tf.keras.utils.to_categorical(labels, NUM_CLASSES)
    # pred_cat = model.predict(test_images_)
    pred_cat = y_pred
    pred = pred_cat.argmax(axis=1)

    performance_metrics = {}
    performance_metrics["accuracy"] = round(accuracy_score(labels, pred), 4)
    performance_metrics["top_5_accuracy"] = round(top_k_categorical_accuracy(labels_cat, pred_cat, k=5).numpy().sum() / len(y), 4)
    performance_metrics["f1_score"] = round(f1_score(labels, pred, average="macro"), 4)
    performance_metrics["precision"] = round(precision_score(labels, pred, average="macro"), 4)
    performance_metrics["recall"] = round(recall_score(labels, pred, average="macro"), 4)
    performance_metrics["loss"] = round(log_loss(labels_cat, pred_cat), 4)

    performance_df.loc[model.name] = performance_metrics
    if verbose:
        return performance_df.loc[model.name]

performance_df = pd.DataFrame(columns=["accuracy", "top_5_accuracy", "precision", "recall", "f1_score", "loss"])


# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D, BatchNormalization, Activation
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

# %%
def get_model_backbone(input_shape, num_classes):
    model = InceptionResNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
    model.trainable = False
    return model

model_backbone = get_model_backbone(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

# %%
model = Sequential(name="InceptionResNetV2")
model.add(model_backbone)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.3))
model.add(Dense(NUM_CLASSES, activation="softmax"))

# %%
model.compile(
    optimizer=Adam(
        learning_rate=LEARNING_RATE
        ),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        'top_k_categorical_accuracy',
        ]
    )

model.summary()

# %%
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, CSVLogger

monitor_metric = 'val_accuracy'
learning_rate_decay_rate = 0.8
model_checkpoint_path = str(MODEL_PATH / f"{model.name}.h5")

def get_callbacks():
    callbacks = {}

    callbacks["EarlyStopping"] = EarlyStopping(
            monitor=monitor_metric,
            patience=5,
            mode = "auto",
            verbose=1,
        )

    callbacks["LearningRateScheduler"] = LearningRateScheduler(step_decay)

    callbacks["ModelCheckpoint"] = ModelCheckpoint(
            model_checkpoint_path,
            monitor=monitor_metric,
            save_best_only=True,
            mode='auto',
            verbose=1,
    )

    return callbacks


def step_decay(epoch):
    initial_lr = LEARNING_RATE
    k = learning_rate_decay_rate
    lr = initial_lr * np.exp(-k*epoch)
    return lr

callbacks = get_callbacks()

# %%
from tensorflow.keras.models import load_model

train_model = not (os.path.exists(MODEL_PATH / f"{model.name}.h5")) or TRAIN_MODELS
steps = round(len(train_images) / 1.25)
if train_model:
    history = model.fit(train_images,
                        validation_data=val_images,
                        epochs=35,
                        steps_per_epoch=steps,
                        callbacks=callbacks,
    )
else:
    model_path = MODEL_PATH / f"{model.name}.h5"
    model = load_model(model_path)
    print(f"{model.name} model loaded from {model_path}")


# %%
if train_model:
    model.evaluate(test_images)
    plot_model_performance(history)

# %%
test_images.shuffle = False
test_labels = test_images.labels
test_labels_pred_ohe = model.predict(test_images)
test_labels_pred = test_labels_pred_ohe.argmax(axis=1)
compute_performance_metrics(test_labels, test_labels_pred_ohe, True)

# %%
performance_df.sort_values(by="accuracy", ascending=False)

# %%
print("""
accuracy          0.8533
top_5_accuracy    0.9818
precision         0.8571
recall            0.8522
f1_score          0.8520
loss              0.4987
Name: InceptionResNetV2, dtype: float64
""")

# %%
# load model
from tensorflow.keras.models import load_model
model = load_model(MODEL_PATH / f"{model.name}.h5")


model.evaluate(test_images)

# %%



