FROM python:3.9-slim-buster

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*
    

RUN apt install gunicorn3 -y

COPY requirements.txt requirements.txt

COPY requirements.txt .
RUN python3.9 -m pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel
RUN python3.9 -m pip install --no-cache-dir \
    -r requirements.txt
    
COPY Object_counter /opt/

RUN pip install -r requirements.txt
RUN python3.9 -m pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel
RUN python3.9 -m pip install --no-cache-dir \
    -r requirements.txt
    
WORKDIR /opt/

CMD ["gunicorn3" ,"-b","0.0.0.0:8000","app:app" ,"--workers = 5"]
