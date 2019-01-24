from abc import ABC,abstractmethod
from tacotron2.train import train
from tacotron2.hparams import create_hparams
from tacotron2.train import load_model
from tacotron2.text import text_to_sequence
import numpy as np
import pyaudio
import wave

import sys
import os


import argparse
import torch


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

class SpeechSynthesis(AbstractClass):

    def __init__(self):
        pass

    def train_mel(self,outdir,logdir,checkpoint):
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

    def train_wav(self):


    def inference_mel(self, checkpoint_path):
        """"
        Performs conversion from text to mel spectogram
        """
        hparams = create_hparams("distributed_run=False,mask_padding=False")
        hparams.sampling_rate = 22050
        hparams.filter_length = 1024
        hparams.hop_length = 256
        hparams.win_length = 1024

        checkpoint_path = "tacotron2/output/checkpoint_1000"
        model = load_model(hparams)
        try:
            model = model.module
        except:
            pass

        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(checkpoint_path)['state_dict'].items()})
        _ = model.eval()

        text = "Fake it till you make it!"
        sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        mel = torch.autograd.Variable(mel_outputs_postnet)
        mel = mel.reshape(80, mel.shape[2])
        mel = mel.data

        filename = 'mel/text_to_mel.pt'
        mel = torch.save(mel, filename)

    def inference_audio(self):
        pass
    def play_audio(self, fname):

        wf = wave.open(fname, 'rb')
        p = pyaudio.PyAudio()

        chunk = 1024

        # open stream based on the wave object which has been input.
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        # read data (based on the chunk size)
        data = wf.readframes(chunk)

        # play stream (looping from beginning of file to the end)
        while data != '':
            # writing to the stream is what *actually* plays the sound.
            stream.write(data)
            data = wf.readframes(chunk)

            # cleanup stuff.
        stream.close()
        p.terminate()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mel', type=bool, help='True or False argument to start training tacotron2 model')
    parser.add_argument('--train_wav',type=bool, help='Argument to train mel spectogram to audio model')
    parser.add_argument('-i','--infer', type=bool, help='True or False to infer text to mel spectogram')
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints', default="./output")
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs',default="./logdir")
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')

    args = parser.parse_args()


    sp=SpeechSynthesis()
    if args.train_mel:
        sp.train_mel(args.output_directory,args.log_directory,args.checkpoint_path)
    if args.train_wave:
        sp.train_wav(**train_config)
    if args.infer:
        sp.inference_mel(args.checkpoint_path)





