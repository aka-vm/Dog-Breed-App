FROM python:3.9-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN mkdir model-binaries
COPY model-binaries/InceptionResNetV2.h5 model-binaries/InceptionResNetV2.h5
COPY model-binaries/breeds_dict.json model-binaries/breeds_dict.json

COPY backend ./backend
COPY static ./static

EXPOSE 8000
CMD python backend/main.py
