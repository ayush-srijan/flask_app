FROM ubuntu:18.04
LABEL maintainer ="ayush.patel@srijan.net"

RUN apt-get -y update
RUN apt-get -y install python3-pip
RUN apt-get -y install gunicorn3

COPY requirements.txt requirements.txt
COPY Object_counter /opt/

RUN pip3 install -r requirements.txt
WORKDIR /opt/

CMD ["gunicorn3" ,"-b","0.0.0.0:8000","app:app" ,"--workers = 5"]
