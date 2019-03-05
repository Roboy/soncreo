# Dockerfile
FROM sunitag/soncreo

## Install Driver Nvidia
WORKDIR /home/tmp


## Install Cuda


## Install CuDnn


## Copying Git Code from Soncreo
RUN git clone https://github.com/Roboy/soncreo.git
RUN git submodule init
RUN git submodule update
