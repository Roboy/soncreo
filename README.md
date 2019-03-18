# Roboy Soncreo
Roboy Soncreo (from Lat. *sonus* - sound and *creō* - I create, make, produce) - a library for Speech Generation based on Deep Learning models.

A pytorch implementaton that combines [Tacotron2] and [NV-Wavenet] to provide audio synthesis from text. It also supports interfacing using ROS2 (not implemented yet)

## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN (tested for version CUDA 9.0 and above)
2. GPU architecture >= smi 61. Check your GPU architecture [here](https://developer.nvidia.com/cuda-gpus)
3. [Pytorch 1.0]
4. (Optional) ROS2 bouncy (for communication via ROS2)

## Setup - Installation of Prerequisites
You can install the Prerequisites on your own or use our Docker Instruction which provides you with a fully installed system on Docker.
### Option 1:  Own Installation
Setup your system by installing current GPU driver, Cuda and cuDNN. Afterwards install the dependencies.
```
pip install -r requirements.txt
```
### Option 2: Docker
A installed Docker with Nvidia environment is a requirement (Installation instruction can be found in `docker/README`). Our Docker contain all the package requirements, ROS2, Cuda 9.0 and cuDNN. Our Docker image can be downloaded via `docker pull sharcc92/soncreo:latest`. The argument `-v /path/to/soncreo/folder:path/to/guest/folder` creates a shared folder between guest and host. You can add the soncreo repo and the provided tacotron2 and nv-wavenet models. 
```
sudo docker run --rm --runtime=nvidia -v /path/to/home/folder:path/to/soncreo/folder -ti sharcc92/soncreo:latest bash
```
But instead of downloading the docker image we recommend to setup your own docker via our provided dockerfile. 
The installation instruction can be found in the README-file in the folder `docker`.

## Setup - Soncreo Repo
1. Clone this repo: `git clone https://github.com/Roboy/soncreo`
2. Initialize submodules: `git submodule init; git submodule update`
3. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/)
### To build nv-wavenet wrapper for pytorch
1. `cd nv-wavenet\pytorch`.
2. Update the ``Makefile`` with the appropriate ``ARCH=sm_70``. Find your ARCH here: https://developer.nvidia.com/cuda-gpus. For example, NVIDIA Titan V has 7.0 compute capability; therefore, it's correct ``ARCH`` parameter is ``sm_70``.
3. Build nv-wavenet and C-wrapper: `make`
4. Install the PyTorch extension: `python build.py install`

## Training Tacotron2
1. `cd tacotron2` and  then update .wav paths: `sed -i -- 's,DUMMY,ljs_dataset_folder/wavs,g' filelists/*.txt`
2. In hparams update `training_files='tacotron2/filelists/ljs_audio_text_train_filelist.txt'` and `validation_files='tacotron2/filelists/ljs_audio_text_val_filelist.txt'` 
3. cd into parent Soncreo directory `cd ..`
4. `python interface.py --output_directory=output --log_directory=logdir`
5. (OPTIONAL) `tensorboard --logdir=output/logdir`

## Training NV-Wavenet
Make a list of the file names to use for training/testing \
  `ls ljs_datset_folder/*.wav | tail -n+10 > train_files.txt`  \
  `ls ljs_dataset_folder/*.wav | head -n10 > test_files.txt`  \
Train the model \
  `python interface_wavenet.py -c nv-wavenet/pytorch/config.json`

## Inference Text to Speech

#### To play audio from text
Add paths for output directory and checkpoints in the config.json file. Then run the following command\
 `python combine.py`
   
#### To infer with our pretrained models for tacotron2 and wavenet
1. Download pretrained models [here](https://drive.google.com/drive/folders/1kwyITQMFvBaQaFTihTQ8DrL_CcVeFaRh?usp=sharing) 
2. Add the paths of the pretrained models in "checkpoint_tac' and 'checkpoint_wav' in config.json
4. Run the following command: `python combine.py`


## (Optional) Connect the Text to Speech Inference via ROS2
This repo contains a ROS2 Server (rospy client library) allows a ROS2 node to communicate.
1. Starting the ros service: `python3 TTS_srv.py`
2. Call the service via a client (simple example client for Roboy is [Pyroboy])
   Run the following from the client \
   `import pyroboy` \
   `pyroboy.say("Input text string")` 




[Pytorch 1.0]: https://github.com/pytorch/pytorch#installation
[Tacotron2]: https://github.com/NVIDIA/tacotron2
[NV-Wavenet]: https://github.com/NVIDIA/nv-wavenet/
[Pyroboy]: https://github.com/Roboy/pyroboy
