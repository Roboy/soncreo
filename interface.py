from abc import ABC,abstractmethod
from tacotron2.train import train
from tacotron2.hparams import create_hparams

import sys
import os

os.chdir('./tacotron2')
import argparse
import torch


class AbstractClass(ABC):

    @abstractmethod
    def train(self):
        pass
    @abstractmethod
    def inference(self):
        pass
    @abstractmethod
    def preprocess(self):
        pass

class Tacotron(AbstractClass):

    def __init__(self):
        pass

    def train(self,outdir,logdir,checkpoint):
        args = parser.parse_args()
        hparams = create_hparams()

        torch.backends.cudnn.enabled = hparams.cudnn_enabled
        torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

        print("FP16 Run:", hparams.fp16_run)
        print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
        print("Distributed Run:", hparams.distributed_run)
        print("cuDNN Enabled:", hparams.cudnn_enabled)
        print("cuDNN Benchmark:", hparams.cudnn_benchmark)

        train(outdir, logdir, checkpoint,
              False, 1, 0, False, hparams)
        pass

    def inference(self):
        pass

    def preprocess(self):
        pass

class DeepVoice(AbstractClass):
    def train(self):
        pass
    def inference(self):
        pass
    def preprocess(self):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints', default="./output")
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs',default="./log")
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')

    args = parser.parse_args()


    Taco=Tacotron()
    Taco.train(args.output_directory,args.log_directory,args.checkpoint_path)

    DP=DeepVoice()

