FROM python:3.6-slim

RUN apt-get update
RUN apt-get install -y wget


RUN apt-get install -y git build-essential libopenblas-dev python3-pip
RUN pip3 install -U pip

RUN wget https://mxnet-public.s3.us-east-2.amazonaws.com/install/jetson/1.6.0/mxnet_cu102-1.6.0-py2.py3-none-linux_aarch64.whl
RUN pip3 install mxnet_cu102-1.6.0-py2.py3-none-linux_aarch64.whl

RUN apt-get install -y python3-scipy python3-pil python3-matplotlib
RUN apt autoremove -y
RUN pip3 install gluoncv

RUN mkdir -p /home/app

COPY . /home/app
