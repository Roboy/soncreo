import os
import subprocess
# import torchaudio


class PostProcessor(object):

    def process(self, wav):
        path = wav + ".0.wav"
        cmd = ['sox', wav, path, "speed", "1.2"]
        subprocess.call(cmd)
        cmd = ['sox', path, wav, "pitch", "175"]
        subprocess.call(cmd)
        os.remove(path)
        path = wav
        return path


if __name__ == "__main__":
    postprocess = PostProcessor()
    postprocess.process("output/text_to_mel.wav")
