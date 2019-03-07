Requirements: Install Docker, location where Dockerfile ist, 

Tested environment: Ubuntu 16.04

## Install NVIDIA Container Runtime for Docker 
Install NVIDIA Container Runtime for Docker [Nvidia-Docker] on your home system
### Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

### Install nvidia-docker2 and reload the Docker daemon configuration
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd

## Build a new docker 
Build a docker with the `dockerfile` provided. Execute in the folder where the `dockerfile` is. Replace `new-local-docker-name` with a name of your choice (e.g. `ubuntucuda:latest`).
```
sudo docker build --no-cache -t new-local-docker-name .
```

## Run the docker 
Run docker with `--runtime=nvidia` and shared folder '-v' (e.g. `sudo docker run --rm --runtime=nvidia -v /home/roboy/3M/soncreo:/tts  -ti ubuntucu9:v1 bash`)
```
sudo docker run --rm --runtime=nvidia -v /path/to/home/folder:path/to/guest/folder -ti new-local-docker-name bash
```

## Useful commands
- check existing dockers on your system: `docker images`

[Nvidia-Docker]: https://github.com/NVIDIA/nvidia-docker
