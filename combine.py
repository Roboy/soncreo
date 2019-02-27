
import argparse
import json
import sys
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
    def inference_audio(self):
        pass
    def play_audio(self):
        pass

if __name__ == "__main__":
    c=Comb()
    #c.train_mel()
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_wav', type=bool, help='Argument to train mel spectogram to audio model', default=True)
    parser.add_argument('--config', type=str,
                        help='JSON file for nv-wavenet configuration', default='./nv-wavenet/pytorch/config.json')

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