# Roboy Soncreo
Roboy Soncreo (from Lat. *sonus* - sound and *creō* - I create, make, produce) - a library for Speech Generation based on Deep Learning models.

A pytorch implementaton that combines [Tacotron2] and [NV-Wavenet] to provide audio synthesis from text. It also supports interfacing using ROS2 (not implemented yet)

## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN
2. [Pytorch 1.0]

## Setup
1. Clone this repo: `git clone https://github.com/Roboy/soncreo`
2. Initialize submodules: `git submodule init; git submodule update`
3. Go to the [Tacotron2] and [NV-Wavenet] repositories and install requirements and datasets as mentioned there.

## Training Tacotron2
1. `python interface.py --output_directory=outdir --log_directory=logdir`
2. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Training NV-Wavenet
  `python interface_wavenet.py -c nv-wavenet/pytorch/config.json`

## Inference Text to Speech (in progress)

#To play audio from text
   `python combine.py --default=False --text='Write your text here' --checkpoint_tac='checkpoint/tac' --checkpoint_wav='checkpoints/wav' --batch=1 output_directory='./output    implementation="persistent"`
   
#To infer with our pretrained models for tacotron2 and wavenet
1. Create a folder named checkpoint and copy tacotron2 and wavenet pretrained models
   `mkdir checkpoints`
2. Create a folder called output (used to save the produced wav file
     `mkdir outputs`
3.Run the following command
    `python combine.py --default=True --text="Write your text here"`




[Pytorch 1.0]: https://github.com/pytorch/pytorch#installation
[Tacotron2]: https://github.com/NVIDIA/tacotron2
[NV-Wavenet]: https://github.com/NVIDIA/nv-wavenet/     


