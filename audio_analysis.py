import math
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fft
import numpy as np

samplerate, samples = wavfile.read('./string_vln.b3.wav')


def get_freq(freq):
    num_samples = samplerate/10
    f, t, Pxx = signal.spectrogram(samples,samplerate,signal.get_window('hamming',int(num_samples)))
    print(f)
    frequency_index = math.floor(freq/10) + 1
    print(f[frequency_index])
    return Pxx[frequency_index + 1,:]


freq = 144
plt.ylabel('Amplitude [db]')
plt.xlabel('Time [sec]')
plt.title('Frequency=' + str(freq) + 'Hz')
plt.plot(get_freq(freq))
plt.show()


