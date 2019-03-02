
import argparse
import json
import torch
import sys
import imp
import wave
import pyaudio
import os

from abc import ABC,abstractmethod

class AbstractClass(ABC):

    @abstractmethod
    def train_mel(self):
        pass
    @abstractmethod
    def train_wav(self):
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

    def inference_audio(self, text, tac_model,wav_model, outdir, batch, implementation):

        from interface import inference_mel
        mel = inference_mel(text, tac_model)
        print(mel)

        from interface_wavenet import infer_wav
        infer_wav(mel, wav_model, outdir, batch, implementation)

        fname = os.path.join(outdir, os.path.splitext(mel)[0] + "." + "wav")
        self.play_audio(fname)


    def play_audio(self,fname):
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
        while len(data) > 0 :
            # writing to the stream is what *actually* plays the sound.
            stream.write(data)
            data = wf.readframes(chunk)

            # cleanup stuff.
        stream.stop_stream()
        stream.close()
        p.terminate()

        print("Output wave generated")

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
    parser.add_argument('--checkpoint_tac', type=str, default="./checkpoints/tacotron2_statedict.pt",
                        required=False, help='checkpoint path')

    parser.add_argument('--checkpoint_wav', default='./checkpoints/wavenet_450000')
    parser.add_argument('-b', '--batch_size', default=1)
    parser.add_argument('-i', '--implementation', type=str, default="auto",
                        help="""Which implementation of NV-WaveNet to use.
                               Takes values of single, dual, or persistent""")

    args = parser.parse_args()

    if args.train_wav:
        c.train_wav(train_config)

    if args.infer_sp:
        text = "Why are Robots shy? Because they have hardware and software but no underware!"
        c.inference_audio(text,args.checkpoint_tac, args.checkpoint_wav,args.output_directory,args.batch_size, args.implementation)