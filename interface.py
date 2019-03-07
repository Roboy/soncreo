import sys
sys.path.insert(0,'./tacotron2')
import numpy as np
import torch
import argparse

from tacotron2.train import train
from tacotron2.hparams import create_hparams
from tacotron2.train import load_model
from tacotron2.text import text_to_sequence

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

def inference_mel(text,checkpoint_path):
    """"
    Performs conversion from text to mel spectogram
    """
    hparams = create_hparams("distributed_run=False,mask_padding=False")
    hparams.sampling_rate = 22050
    hparams.filter_length = 1024
    hparams.hop_length = 256
    hparams.win_length = 1024

    model = load_model(hparams)
    try:
        model = model.module
    except:
        pass

    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(checkpoint_path)['state_dict'].items()})
    _ = model.eval()

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')


    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train_mel(args.output_directory, args.log_directory, args.checkpoint_path)







