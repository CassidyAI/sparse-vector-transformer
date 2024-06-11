FROM python:3.8-slim-buster
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ARG PORT=80
EXPOSE $PORT
ENTRYPOINT uvicorn main:app --host 0.0.0.0 --port $PORT
