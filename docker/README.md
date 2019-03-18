# Create your own Soncreo Docker

Tested Host environment: Ubuntu 16.04, GPU Geforce 1080 Ti

## Install Docker
The installation instruction is inspired from [Docker for Ubuntu].

### Preinstallation for Docker
First, in order to ensure the downloads are valid, add the GPG key for the official Docker repository to your system:
```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```
Add the Docker repository to APT sources:
```
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
```
Next, update the package database with the Docker packages from the newly added repo:
```
sudo apt-get update
```
Make sure you are about to install from the Docker repo instead of the default Ubuntu 16.04 repo:
```
apt-cache policy docker-ce
```
You should see output similar to the follow:
```
Output of apt-cache policy docker-ce
docker-ce:
  Installed: (none)
  Candidate: 18.06.1~ce~3-0~ubuntu
  Version table:
     18.06.1~ce~3-0~ubuntu 500
        500 https://download.docker.com/linux/ubuntu xenial/stable amd64 Packages
```
Notice that docker-ce is not installed, but the candidate for installation is from the Docker repository for Ubuntu 16.04 (xenial).
### Install Docker
Finally, install Docker:
```
sudo apt-get install -y docker-ce
```
## Install NVIDIA Container Runtime for Docker 
Install NVIDIA Container Runtime for Docker [Nvidia-Docker] on your home system

### If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
```
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo apt-get purge -y nvidia-docker
```
### Add the package repositories
```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
```

### Install nvidia-docker2 and reload the Docker daemon configuration
```
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP docker
```

## Build a new docker 
Build a docker with the `dockerfile` provided. Execute in the folder where the `dockerfile` is (`cd docker`). Replace `new-local-docker-name` with a name of your choice (e.g. `sharcc92/soncreo:latest`).
```
sudo docker build --no-cache -t new-local-docker-name .
```

## Run the docker 
Run docker with `--runtime=nvidia` and shared folder `-v` and audio output device `--device` and same network `--network=host` (e.g. `sudo docker run --rm --runtime=nvidia -v /home/roboy/3M/soncreo:/tts --network=host --device /dev/snd -ti ubuntucu9:v1 bash`)
```
sudo docker run --rm --runtime=nvidia -v /path/to/home/folder:path/to/guest/folder -ti new-local-docker-name bash
```
Notes: With the shared folder `-v` you can share your soncreo repo or exchange the tacotron2 and nv-wavenet. With audio output device `--device /dev/snd` you can play audio from the image in your home system. The argument network `--network=host` makes sure that you are in the same network as your home system to find interface via ROS2.

## Useful commands
- check existing dockers on your system: `docker images`
- check history of a docker: `docker image history IMAGE`

[Nvidia-Docker]: https://github.com/NVIDIA/nvidia-docker
[Docker for Ubuntu]: https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-16-04

