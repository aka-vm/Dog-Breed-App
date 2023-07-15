# Dog-Breed-App
[![Docker Pulls](https://badgen.net/docker/pulls/akavm/dog-breed-app?icon=docker&label=pulls)](https://hub.docker.com/repository/docker/akavm/dog-breed-app/)
[![Docker Image Size](https://badgen.net/docker/size/akavm/dog-breed-app/0.1.1/arm64?icon=docker&label=linux/arm64)](https://hub.docker.com/layers/akavm/dog-breed-app/0.1.1/images/sha256-c9e16a306518830e23b4565171b193abd62d4892f11a119478c6efa131f5ceb7?)
[![Docker Image Size](https://badgen.net/docker/size/akavm/dog-breed-app/0.1.1/amd64?icon=docker&label=linux/amd64)](https://hub.docker.com/layers/akavm/dog-breed-app/0.1-amd64/images/sha256-c9e16a306518830e23b4565171b193abd62d4892f11a119478c6efa131f5ceb7)


This Repository contains a Web App along with the model nb that can be used to predict the breed of a dog based on the image uploaded.

## Model
[This Notebook](models/InceptionResNetV2.ipynb) trains the model on Stanford Dogs Dataset.
[This Repo](https://github.com/aka-vm/Hello-CV/tree/master/Stanford%20Dogs) contains the original code, but some modifications are made to make it better.

## Web App
The server uses [FastAPI](https://fastapi.tiangolo.com/).<br>
I've hosted the app on Azure and DigitalOcean, and it works fine. But I recommend using [ngrok](https://ngrok.com/) for quick testing. The Docker image is also available on [Docker Hub](https://hub.docker.com/repository/docker/akavm/dog-breed-app).


## Running the app
Two ways to run the app, If you want a quick test, I recommend using Docker. It'll work on arm64 and amd64.
### Using Regular Python

1. Clone the repo.
2. create a virtual environment and install the requirements from `requirements.py`.
3. Download the model from [Google Drive](https://drive.google.com/file/d/1hH6c4YDjSQ9F2FV1p1QFuHnJ1ouKf_vQ/view?usp=share_link) or `releases` and place it in the `model-binaries` folder.
4. Run the app using `python backend/main.py`. The app will be hosted on port [8000](localhost:8000).

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
wget -O model-binaries/InceptionResNetV2.h5 https://github.com/aka-vm/Dog-Breed-App/releases/download/Classification-Model/InceptionResNetV2.h5
# Run server
python backend/main.py
# for web testing I recommend using ngrok
```
### Using Docker
```bash
# Previsouly I built two images for arm64 and amd64, but now I'm using multi-arch build
docker pull akavm/dog-breed-app:0.1.1
docker run -p 8000:8000 akavm/dog-breed-app:0.1.1
```


----------


Mobile View:
<!-- Height 200px -->
<img src="GIFs/Mobile%20View.gif" alt="Mobile View" width="40%">

----------

PC View:

![](/GIFs/PC%20View.gif)
