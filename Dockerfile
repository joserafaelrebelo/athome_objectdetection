FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3

RUN apt-get update
RUN apt-get install -y \
                        build-essential \
                        git \
                        libopenblas-dev \
                        libopencv-dev \
                        python3-pip \
                        python-numpy

RUN pip3 install --upgrade \
                        pip \
                        setuptools \
                        numpy


RUN git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet


ENV export PATH=/usr/local/cuda/bin:$PATH
ENV export MXNET_HOME=$HOME/mxnet/
ENV export PYTHONPATH=$MXNET_HOME/python:$PYTHONPATH

$MXNET_HOME/ci/build.py -p jetson

RUN apt-get install -y python3-scipy python3-pil python3-matplotlib
RUN apt autoremove -y
RUN pip3 install gluoncv

RUN mkdir -p /home/app

COPY . /home/app
