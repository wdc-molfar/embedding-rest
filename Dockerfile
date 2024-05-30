FROM python:3.10

# Set the working directory in the container

RUN apt install git -y

ENV NODE_ENV=production
WORKDIR /data
COPY . .

RUN apt-get update -qq \
    && apt-get install -qq -y --no-install-recommends \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel

RUN python3 -m pip install --no-cache-dir \
    -r requirements.txt

RUN mkdir -p ./models

RUN python3 ./src/models.py
RUN uvicorn src.main:app --reload

# prevent downloading models on every restart
VOLUME ["./src/models"]
