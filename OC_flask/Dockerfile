FROM ubuntu:22.04
LABEL maintainer ="ayush.patel@srijan.net"

RUN apt-get update -y
RUN apt-get install python3-pip -y
RUN apt-get install gunicorn3 -y
RUN apt-get -y install python3-torch
RUN apt-get -y install python3-torchvision
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install -y opencv-python-headless
RUN pip install opencv-python-headless

COPY requirements.txt requirements.txt
COPY Object_counter /opt/

RUN pip3 install -r requirements.txt
WORKDIR /opt/

CMD ["gunicorn3" ,"-b","0.0.0.0:8000","app:app" ,"--workers = 5"]
