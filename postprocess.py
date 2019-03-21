import wave
import numpy as np
import librosa

from ctypes import byref, c_float, c_void_p, CDLL


class RNNoise(object):
    def __init__(self):
        self._native = CDLL('./librnnoise.so')
        self._native.rnnoise_process_frame.restype = c_float
        self._native.rnnoise_process_frame.argtypes = (
            c_void_p, c_void_p, c_void_p)
        self._native.rnnoise_create.restype = c_void_p
        self._handle = self._native.rnnoise_create()
        self._buf = (c_float * self.frame_size)()

    @property
    def frame_size(self):
        return 480

    def process_frame(self, samples):
        if len(samples) > self.frame_size:
            raise ValueError
        for i in range(len(samples)):
            self._buf[i] = samples[i]
        for i in range(len(samples), self.frame_size):
            self._buf[i] = 0
        vad_prob = self._native.rnnoise_process_frame(
            self._handle, byref(self._buf), byref(self._buf))
        for i in range(len(samples)):
            samples[i] = self._buf[i]
        return vad_prob

    def __del__(self):
        if self._handle:
            self._native.rnnoise_destroy(self._handle)

# Created input file with:
# mpg123  -w 20130509talk.wav 20130509talk.mp3
wr = wave.open('text_to_mel.wav', 'r')
par = list(wr.getparams()) # Get the parameters from the input.
# This file is stereo, 2 bytes/sample, 44.1 kHz.
par[3] = 0  # The number of samples will be set by writeframes.

# Open the output file
ww = wave.open('text_to_mel_no_noise.wav', 'w')
ww.setparams(tuple(par))  # Use the same parameters as the input file.

lowpass = 200  # Remove lower frequencies.
highpass = 6000  # Remove higher frequencies.

sz = wr.getframerate() # Read and process 1 second at a time.
c = int(wr.getnframes()/sz) # whole file
for num in range(c):
    print('Processing {}/{} s'.format(num+1, c))
    da = np.frombuffer(wr.readframes(sz), dtype=np.int16)
    left, right = da[0::2], da[1::2] # left and right channel
    lf, rf = np.fft.rfft(left), np.fft.rfft(right)
    lf[:lowpass], rf[:lowpass] = 0, 0  # low pass filter
    lf[55:66], rf[55:66] = 0, 0  # line noise
    lf[highpass:], rf[highpass:] = 0, 0  # high pass filter
    nl, nr = np.fft.irfft(lf), np.fft.irfft(rf)
    ns = np.column_stack((nl, nr)).ravel().astype(np.int16)
    ww.writeframes(ns.tostring())
# Close the files.
wr.close()
ww.close()

wav, sr = librosa.load('text_to_mel.wav')
wav = librosa.resample(y, sr, 48000)
wav = librosa.effects.time_stretch(wav, 1.1)
wav = librosa.effects.pitch_shift(wav, sr, n_steps=3.0)
librosa.output.write_wav('text_to_mel_no_noise.wav', wav, sr)

