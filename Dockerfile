FROM python:3.9.13
WORKDIR /app

COPY requirements.txt .
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir model-binaries
RUN wget -O /app/model-binaries/InceptionResNetV2.h5 https://github.com/aka-vm/Dog-Breed-App/releases/download/Classification-Model/InceptionResNetV2.h5

COPY backend ./backend
COPY static ./static
COPY model-binaries/breeds_dict.json model-binaries/breeds_dict.json

EXPOSE 8000
CMD python backend/main.py
