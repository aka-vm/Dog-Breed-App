# Dog-Breed-App
This Repository contains a Web App along with the model nb that can be used to predict the breed of a dog based on the image uploaded.

## Model
[This Notebook](models/InceptionResNetV2.ipynb) trains the model on Stanford Dogs Dataset.
[This Repo](https://github.com/aka-vm/Hello-CV/tree/master/Stanford%20Dogs) contains the original code, but some modifications are made to make it better.

## Web App
Hosted using [FastAPI](https://fastapi.tiangolo.com/) on [Azure Virtual Machines](https://azure.microsoft.com/en-us/services/virtual-machines/).<br>
Click [<u>**here**</u>](http://20.219.1.85:8000) to use the app.<br>
Note: This may not work, but you can see the GIF below to have an idea of the app.

## Running the app
### Locally
locally:
1. Clone the repo.
2. create a virtual environment and install the requirements from `requirements.py`.
3. Download the model from [Google Drive](https://drive.google.com/file/d/1hH6c4YDjSQ9F2FV1p1QFuHnJ1ouKf_vQ/view?usp=share_link) or `releases` and place it in the `model-binaries` folder.
4. Run the app using `uvicorn backend.main:app`. The app will be hosted on port [8000](localhost:8000).

```bash
# clone
git clone https://github.com/aka-vm/Dog-Breed-App
cd Dog-Breed-App

# Dependencies and environment
pip install virtualenv
virtualenv --python=python3.9.12 venv
source venv/bin/activate
pip install -r requirements.txt

# Download The model and paste it in the model-binaries folder

# Run server
uvicorn backend.main:app --reload
# for web testing I recommend using ngrok
```


----------


Mobile View:
<!-- Height 200px -->
<img src="GIFs/Mobile%20View.gif" alt="Mobile View" width="40%">

----------

PC View:

![](/GIFs/PC%20View.gif)
