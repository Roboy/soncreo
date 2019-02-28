
import argparse
import json
import torch
import sys
import os
#from interface_wavenet import train_wav as nv
#from interface import train_mel as tac2
from abc import ABC,abstractmethod

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

class Comb(AbstractClass):
    def __init__(self):
        pass
    def train_mel(self):
        #sys.path.insert(0, './tacotron2')
        from interface import train_mel as tac2
        tac2('./outdir','./logdir',None)
    def train_wav(self, train_config):
        #sys.path.insert(0, './nvwavenet/pytorch')
        from interface_wavenet import train_wav as nv
        nv(train_config)
    def inference_mel(self):
        pass
    def inference_audio(self, tac_model,wav_model, outdir, batch):
        from interface import inference_mel
        text="Why are robots shy? Because they have hardware and software but no underwear!"
        mel = inference_mel(tac_model)

        filename = 'mel/text_to_mel.pt'
        mel = torch.save(mel, filename)


        sys.path.insert(0, './nv-wavenet/pytorch')

        print(os.getcwd())
        print(sys.path)

        from interface_wavenet import infer_wav


        infer_wav(filename, wav_model, outdir, batch)


    def play_audio(self):
        pass

if __name__ == "__main__":
    c=Comb()
    #c.train_mel()
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_wav', type=bool, help='Argument to train mel spectogram to audio model', default=False)
    parser.add_argument('--infer_sp', type=bool, help='Argument to infer speech from text', default=True)
    parser.add_argument('--config', type=str,
                        help='JSON file for nv-wavenet configuration', default='./nv-wavenet/pytorch/config.json')

    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save audio files', default="./output")
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs', default="./logdir")
    parser.add_argument('--checkpoint_tac', type=str, default="./checkpoints/checkpoint_0",
                        required=False, help='checkpoint path')

    parser.add_argument('--checkpoint_wav', default='./checkpoints/wavenet_4000')
    parser.add_argument('-b', '--batch_size', default=1)
    parser.add_argument('-i', '--implementation', type=str, default="auto",
                        help="""Which implementation of NV-WaveNet to use.
                               Takes values of single, dual, or persistent""")

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    print(train_config)
    global data_config
    data_config = config["data_config"]
    global dist_config
    dist_config = config["dist_config"]
    global wavenet_config
    wavenet_config = config["wavenet_config"]




    if args.train_wav:
        c.train_wav(train_config)

    if args.infer_sp:
        c.inference_audio(args.checkpoint_tac, args.checkpoint_wav,args.output_directory,args.batch_size)