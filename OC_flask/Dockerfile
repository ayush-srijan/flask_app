FROM python:3.9-slim-buster
LABEL maintainer ="ayush.patel@srijan.net"

RUN apt update && apt install -y
ARG DEBIAN_FRONTEND="noninteractive"

RUN pip install pipenv
RUN pipenv install gunicorn

COPY requirements.txt requirements.txt
COPY Object_counter /opt/

RUN pip install -r requirements.txt
WORKDIR /opt/

CMD ["gunicorn3" ,"-b","0.0.0.0:8000","app:app" ,"--workers = 5"]
