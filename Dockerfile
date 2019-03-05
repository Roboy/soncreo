# Dockerfile
FROM sunitag/soncreo

RUN apt-get update
RUN apt-get upgrade -y

#creating a new directory and copying your apllication inside the container
RUN mkdir -p /tts
WORKDIR /tts
COPY . .

## Install Driver Nvidia



## Install Cuda


## Install CuDnn


## Install Gitclone

