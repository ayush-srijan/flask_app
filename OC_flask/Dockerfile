FROM python:3-alpine
LABEL maintainer ="ayush.patel@srijan.net"

RUN apk update
RUN apk upgrade
RUN apk add py3-pip 
RUN apk add py3-gunicorn 
RUN apk add --no-cache tesseract-ocr python3 py3-numpy && \
    pip3 install --upgrade pip setuptools wheel && \
    pip3 install matplotlib


COPY requirements.txt requirements.txt
COPY Object_counter /opt/

RUN pip3 install -r requirements.txt
WORKDIR /opt/


CMD ["gunicorn3" ,"-b","0.0.0.0:8000","app:app" ,"--workers = 5"]
