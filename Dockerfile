# Super simple example of a Dockerfile
# Dockerfile
FROM missxa/bouncy-roboy

# 
RUN mkdir -p /tts
WORKDIR /tts
COPY .  .

## Install Driver Nvidia
RUN echo "Install Driver Nvidia"
RUN cd tts/
RUN mkdir Downloads/
RUN cd Downloads/
RUN apt-get purge nvidia*
RUN add-apt-repository ppa:graphics-drivers
RUN apt-get update
RUN apt-get install nvidia-410
RUN lsmod | grep nvidia oder nvidia-smi

#install cuda and cudnn


RUN echo "Installing Cuda 10"
RUN apt-get install gcc
RUN dpkg -i cuda-repo-ubuntu1604-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
RUN apt-key add /var/cuda-repo-<version>/7fa2af80.pub
RUN apt-get update
RUN apt-get install cuda

RUN nano ~/.bashrc
RUN export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}} 

RUN echo "Checking driver toolkit and cuda version"
RUN cat /proc/driver/nvidia/version
RUN nvcc -V

RUN echo "installing CuDNN"
RUN dpkg -i libcudnn7_7.5.0.56-1+cuda10.0_amd64.deb
RUN dpkg -i libcudnn7-dev_7.5.0.56-1+cuda10.0_amd64.deb


## Copying Git Code from Soncreo
RUN git clone https://github.com/Roboy/soncreo.git
RUN git submodule init
RUN git submodule update
