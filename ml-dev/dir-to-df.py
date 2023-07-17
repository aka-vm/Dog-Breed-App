# %% [markdown]
# ### Dir to Pandas DataFrame
# Running This Notebook will produce a Pandas DataFrame containing each path of each image in the database along with the breed and annotation of the image.

# %%
import numpy as np
import pandas as pd

import os
import shutil
import pathlib
import sys

# %%
# go to parent directory
os.chdir("..")
import paths

REPO_DIR = paths.get_repo_path()
ROOT_DIR = REPO_DIR / "models"
DATA_BASE_PATH = paths.get_data_path() / "stanford-dogs-dataset"
IMAGES_PATH = DATA_BASE_PATH / "images/Images"
ANNOTATIONS_PATH = DATA_BASE_PATH / "annotations/Annotation"

RANDOM_SEED = 42

# set path to repo_dir
os.chdir(REPO_DIR)

# %%
breed_dir_name = [
        breed
        for breed in sorted(os.listdir(IMAGES_PATH))
        if not breed.startswith(".") and os.path.isdir(IMAGES_PATH / breed)
]

len(breed_dir_name)

# %%
dogs_df = pd.DataFrame(columns=["breed", "image_path", "annotation_path"])

for breed_dir in breed_dir_name:
    breed_name = " ".join(breed_dir.replace("_", "-").split("-")[1:]).title()

    breed_images_dir_path = IMAGES_PATH / breed_dir
    breed_annotations_dir_path = ANNOTATIONS_PATH / breed_dir

    breed_images_name = [
            image
            for image in sorted(os.listdir(breed_images_dir_path))
            if not image.startswith(".") and image.endswith((".jpg", ".jpeg", ".png"))
    ]
    breed_annotations_name = [
            image.split(".")[0]
            for image in breed_images_name
    ]

    breed_images_path = [
            breed_images_dir_path / image
            for image in breed_images_name
            if os.path.isfile(breed_images_dir_path / image)
    ]
    breed_annotations_path = [
            breed_annotations_dir_path / annotation
            for annotation in breed_annotations_name
            if os.path.isfile(breed_annotations_dir_path / annotation)
    ]

    dogs_df = pd.concat([dogs_df, pd.DataFrame({"breed": breed_name, "image_path": breed_images_path, "annotation_path": breed_annotations_path})])


dogs_df

# %%
dogs_df_path = DATA_BASE_PATH / "dogs_df.csv"
dogs_df = dogs_df.sort_values(by=["breed", "image_path"])
dogs_df.to_csv(dogs_df_path, index=False)

dogs_df = pd.read_csv(dogs_df_path)
dogs_df

# %%
import json

breeds = dogs_df["breed"].unique()
breeds_dict = {i: breed for i, breed in enumerate(breeds)}

with open(ROOT_DIR / "breeds_dict.json", "w") as f:
    json.dump(breeds_dict, f, indent=2)

# %%



