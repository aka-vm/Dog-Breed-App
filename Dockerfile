FROM python:3.9.13
WORKDIR /app

COPY requirements.txt .
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY model-binaries ./model-binaries
COPY static ./static

EXPOSE 8000
CMD python backend/main.py
