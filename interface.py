import sys
sys.path.insert(0,'./tacotron2')

print(sys.path)
import os
import json


import argparse
import torch



#from abc import ABC,abstractmethod
from tacotron2.train import train
from tacotron2.hparams import create_hparams
from tacotron2.train import load_model
from tacotron2.text import text_to_sequence



#from nvwavenet.pytorch.train import train as train_wavenet

import numpy as np
import pyaudio
import wave



'''

class AbstractClass(ABC):

    @abstractmethod
    def train_mel(self):
        pass
    @abstractmethod
    def train_wav(self):
        pass
    @abstractmethod
    def inference_mel(self):
        pass
    @abstractmethod
    def inference_audio(self):
        pass
    @abstractmethod
    def play_audio(self):
        pass
'''

#class SpeechSynthesis(AbstractClass):

def __init__(self):
    pass

def train_mel(outdir,logdir,checkpoint):

    hparams = create_hparams()

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(outdir, logdir, checkpoint,
          True, 1, 0, False, hparams)

def train_wav(train_config):

    train_wavenet(1,0,'',**train_config)



def inference_mel(checkpoint_path):
    """"
    Performs conversion from text to mel spectogram
    """
    hparams = create_hparams("distributed_run=False,mask_padding=False")
    hparams.sampling_rate = 22050
    hparams.filter_length = 1024
    hparams.hop_length = 256
    hparams.win_length = 1024

    #checkpoint_path = "tacotron2/output/checkpoint_1000"
    model = load_model(hparams)
    try:
        model = model.module
    except:
        pass

    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(checkpoint_path)['state_dict'].items()})
    _ = model.eval()


    text = "Why are robots shy? Because they have hardware and software but no underwear!"

    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    mel = torch.autograd.Variable(mel_outputs_postnet)
    mel = mel.reshape(80, mel.shape[2])
    mel = mel.data


    filename = "text_to_mel"
    mel = torch.save(mel, filename)

    file = open(str(filename) + ".txt", 'w')
    file.write(filename)
    file.close()

    return file.name

def inference_audio():
    pass
def play_audio():
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mel', type=bool, help='True or False argument to start training tacotron2 model',default=False)
    parser.add_argument('--train_wav',type=bool, help='Argument to train mel spectogram to audio model',default=False)
    parser.add_argument('--config', type=str,
                        help='JSON file for nv-wavenet configuration',default='./nv-wavenet/pytorch/config.json')
    parser.add_argument('-i','--infer_mel', type=bool, default=True, help='True or False to infer text to mel spectogram')
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints', default="./output")
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs',default="./logdir")

    parser.add_argument('-c', '--checkpoint_path', type=str, default='./checkpoints/tacotron2_statedict.pt',

                        required=False, help='checkpoint path')
    args = parser.parse_args()


    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global dist_config
    dist_config = config["dist_config"]
    global wavenet_config
    wavenet_config = config["wavenet_config"]



    if args.train_mel:
        sys.path.append('./tacotron2')
        train_mel(args.output_directory,args.log_directory,args.checkpoint_path)
    if args.train_wav:
        sys.path.append('./nvwavenet/pytorch')
        train_wav(train_config)
    if args.infer_mel:
        inference_mel(args.checkpoint_path)






