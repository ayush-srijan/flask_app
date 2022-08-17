FROM python:3.9-slim-buster

RUN python3 -m venv /opt/venv

RUN apt update && apt install -y
RUN apt install gunicorn3 -y

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*


RUN python3.9 -m pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel

COPY requirements.txt .
RUN . /opt/venv/bin/activate && pip install -r requirements.txt
    
COPY Object_counter /opt/

WORKDIR /opt/

CMD . /opt/venv/bin/activate && gunicorn3 -b 0.0.0.0:8000 app:app --workers 5
