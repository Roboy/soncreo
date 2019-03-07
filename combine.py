import argparse
import wave
import pyaudio
import os
#from logmmse import logmmse_from_file

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
        self.batch = 1
    def train_mel(self):
        #sys.path.insert(0, './tacotron2')
        from interface import train_mel as tac2
        tac2('./outdir', './logdir', None)
    def train_wav(self, train_config):
        #sys.path.insert(0, './nvwavenet/pytorch')
        from interface_wavenet import train_wav as nv
        nv(train_config)

    def load_models(self, tac_model= "./checkpoints/tacotron2_statedict.pt",wav_model='./checkpoints/wavenet_640000'):
        from interface import load_mel_model
        mel_model = load_mel_model(tac_model)
        from interface_wavenet import load_wav_model
        nvwav_model = load_wav_model(wav_model)
        return mel_model,nvwav_model

    def inference_audio(self, text,mel_model,wav_model,outdir="./output", batch=1, implementation="auto"):

        from interface import inference_mel
        mel = inference_mel(text, mel_model)
        print(mel)

        from interface_wavenet import infer_wav
        infer_wav(mel,wav_model[0],wav_model[1], outdir, batch, implementation)

        fname = os.path.join(outdir, os.path.splitext(mel)[0] + "." + "wav")

        #out = logmmse_from_file(fname,output_file="denoised")
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
    c=Comb()
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_wav', type=bool, help='Argument to train mel spectogram to audio model', default=False)
    parser.add_argument('--text', type=str, help='Text input for speech generation', default="The team did some research and was able to get a evaluation license for testing it in my motor cortex also known as the FPGA, but they are still working on integrating it.")


    parser.add_argument('--default_vals', type=bool, help='All arguments are default values', default=True)

    parser.add_argument('--config', type=str,
                        help='JSON file for nv-wavenet configuration', default='./nv-wavenet/pytorch/config.json')

    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save audio files')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs', default="./logdir")
    parser.add_argument('--checkpoint_tac', type=str,
                        required=False, help='checkpoint path')

    parser.add_argument('--checkpoint_wav')
    parser.add_argument('-b', '--batch_size')
    parser.add_argument('-i', '--implementation', type=str,
                        help="""Which implementation of NV-WaveNet to use.
                               Takes values of single, dual, or persistent""")
    #parser.add_argument('--text', type=str, default="1It all started in October 2018 with a group of students.")


    args = parser.parse_args()
    if args.default_vals:
        mel_model,wav_model = c.load_models()
        c.inference_audio(args.text, mel_model, wav_model)
    else:
        mel_model,wav_model = c.load_models(args.checkpoint_tac,args.checkpoint_wav)
        c.inference_audio(args.text, mel_model, wav_model, args.output_directory, args.batch_size, args.implementation)
