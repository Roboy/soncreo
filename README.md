# Roboy Soncreo
Roboy Soncreo (from Lat. *sonus* - sound and *creō* - I create, make, produce) - a library for Speech Generation based on Deep Learning models.

A pytorch implementaton that combines [Tacotron2] and [NV-Wavenet] to provide audio synthesis from text. It also supports interfacing using ROS2 (not implemented yet)

## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN
2. [Pytorch 1.0]

## Setup
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
2. cd into parent Soncreo directory `cd ..`
3. `python interface.py --output_directory=output --log_directory=logdir`
4. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Training NV-Wavenet
Make a list of the file names to use for training/testing \
  `ls ljs_datset_folder/*.wav | tail -n+10 > train_files.txt`  \
  `ls ljs_dataset_folder/*.wav | head -n10 > test_files.txt`  \
Train the model \
  `python interface_wavenet.py -c nv-wavenet/pytorch/config.json`

## Inference Text to Speech

#### To play audio from text
 `python combine.py --default=False --text='Write your text here' --checkpoint_tac='checkpoint/tac' --checkpoint_wav='checkpoints/wav' --batch=1 output_directory='./output --implementation="persistent"`
   
#### To infer with our pretrained models for tacotron2 and wavenet
1. Download pretrained models [here](https://drive.google.com/drive/folders/1kwyITQMFvBaQaFTihTQ8DrL_CcVeFaRh?usp=sharing) 
2. Create a folder named checkpoint and copy tacotron2 and wavenet pretrained models: `mkdir checkpoints`
3. Create a folder called output (used to save the produced wav file: `mkdir outputs`
4. Run the following command: `python combine.py --default=True --text="Write your text here"`


## (Optional) Connect the Text to Speech Inference via ROS2
This repo contains a ROS2 Server (rospy client library) allows a ROS2 node to communicate.
1. Starting the ros service: `python3 TTS_srv.py`
2. Call the service via a client (simple example client for Roboy is [Pyroboy])




[Pytorch 1.0]: https://github.com/pytorch/pytorch#installation
[Tacotron2]: https://github.com/NVIDIA/tacotron2
[NV-Wavenet]: https://github.com/NVIDIA/nv-wavenet/
[Pyroboy]: https://github.com/Roboy/pyroboy
