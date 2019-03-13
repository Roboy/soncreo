import argparse
import wave
import pyaudio
import os
import json
import time

from abc import ABC,abstractmethod
class AbstractClass(ABC):

    @abstractmethod
    def load_models(self):
        pass
    @abstractmethod
    def inference_audio(self):
        pass
    @abstractmethod
    def play_audio(self):
        pass
    @abstractmethod
    def preprocess(self):
        pass

class Comb(AbstractClass):
    def __init__(self, tac_model,wav_model):
        self.mel_model,self.wav_model = self.load_models(tac_model,wav_model)

    def preprocess(self, text):

        if (text[len(text) - 1] == '.' or text[len(text) - 1] == '?' or text[len(text) - 1] == '!'):
            text = text
        else:
            text = text + '.'
        print(text)
        return text

    def load_models(self, tac_model= "./checkpoints/tacotron2_statedict.pt",wav_model='./checkpoints/wavenet_640000'):
        from interface import load_mel_model
        mel_model = load_mel_model(tac_model)
        from interface_wavenet import load_wav_model
        nvwav_model = load_wav_model(wav_model)
        return mel_model,nvwav_model

    def inference_audio(self, text,outdir, batch, implementation):

        text = self.preprocess(text)
        start = time.time()
        from interface import inference_mel
        mel = inference_mel(text, self.mel_model)
        print(mel)

        from interface_wavenet import infer_wav
        infer_wav(mel,self.wav_model[0],self.wav_model[1], outdir, batch, implementation)

        fname = os.path.join(outdir, os.path.splitext(mel)[0] + "." + "wav")

        end = time.time()
        print("Inference time", end-start)
        #out = logmmse_from_file(fname,output_file="denoised")

        start_a =time.time()
        self.play_audio(fname)
        end_a = time.time()
        print("Audio playback time", end_a-start_a)

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
        while len(data) > 0:
            # writing to the stream is what *actually* plays the sound.
            stream.write(data)
            data = wf.readframes(chunk)

            # cleanup stuff.
        stream.stop_stream()
        stream.close()
        p.terminate()

        print("Output wave generated")


if __name__ == "__main__":

    with open('config.json') as json_data_file:
        data = json.load(json_data_file)

    parser = argparse.ArgumentParser()

    parser.add_argument('--text', type=str, help='Text input for speech generation', default=data["text"])
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save audio files', default= data["output_directory"])
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs', default=data["log_directory"])
    parser.add_argument('--checkpoint_tac', type=str,
                        required=False, help='Tacotron2 checkpoint path', default=data["checkpoint_tac"])

    parser.add_argument('--checkpoint_wav', type=str, required=False, help="Wavenet checkpoint path", default=data["checkpoint_wav"])
    parser.add_argument('-b', '--batch_size', help= "batch size for inference", default=data["batch"])
    parser.add_argument('-i', '--implementation', type=str,
                        help="""Which implementation of NV-WaveNet to use.
                               Takes values of single, dual, or persistent""", default=data["implementation"])

    args = parser.parse_args()

    c = Comb(args.checkpoint_tac,args.checkpoint_wav)

    start_t=time.time()
    c.inference_audio(args.text,args.output_directory, args.batch_size, args.implementation)
    end_t=time.time()
    print("Total",end_t-start_t)

