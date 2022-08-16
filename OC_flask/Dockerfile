FROM alpine:latest
MAINTAINER ayush.patel@srijan.net

RUN apt-get install update -y
RUN apt-get install pythn3-pip -y
RUN apt-get install gunicorn -y

COPY requirements.txt requirements.txt
COPY Object_counter /opt/

RUN pip3 install -r requirements.txt
WORKDIR /opt/


CMD ["gunicorn3" ,"-b","0.0.0.0:8000","app:app" ,"--workers=5"]
