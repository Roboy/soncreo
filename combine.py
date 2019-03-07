
import argparse
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
        self.batch = 1
    def train_mel(self):
        #sys.path.insert(0, './tacotron2')
        from interface import train_mel as tac2
        tac2('./outdir', './logdir', None)
    def train_wav(self, train_config):
        #sys.path.insert(0, './nvwavenet/pytorch')
        from interface_wavenet import train_wav as nv
        nv(train_config)


    def inference_audio(self, text, tac_model="./checkpoints/tacotron2_statedict.pt", wav_model='./checkpoints/wavenet_500000',
                        outdir="./output", batch=1, implementation="auto"):
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
        for i in range(p.get_device_count()):
            print(p.get_device_info_by_index(i))

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
    parser.add_argument('--text', type=str, help='Text input for speech generation', default="Fake it till you make it. Fake it till you make it")

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
    parser.add_argument('--text', type=str, default="Hello.")


    args = parser.parse_args()
    if args.default_vals:
        c.inference_audio(args.text)
    else:
        c.inference_audio(args.text, args.checkpoint_tac, args.checkpoint_wav, args.output_directory, args.batch_size, args.implementation)
