# Install dependencies
pip install virtualenv --quiet
if [ ! -d "venv" ]; then
    echo "Creating virtual environment"
    virtualenv --python=python3.9 venv
    echo "Installing dependencies"
    pip install -r requirements.txt --quiet
fi
source venv/bin/activate

# Download the model
clf_model_file_source="https://github.com/aka-vm/Dog-Breed-App/releases/download/Classification-Model/InceptionResNetV2.h5"
if [ ! -d "model-binaries" ]; then
    mkdir model-binaries
fi
if [ ! -f "model-binaries/InceptionResNetV2.h5" ]; then
    echo "Downloading the model"
    wget -O model-binaries/InceptionResNetV2.h5 $clf_model_file_source
fi

# run the app
python backend/main.py
```