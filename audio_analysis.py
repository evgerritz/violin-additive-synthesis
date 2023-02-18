#! /usr/bin/env python
import math
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fft
import numpy as np
import scipy.stats as st


def get_freq(freq, samplerate, samples, smooth=True):
    # good value empirically for maximizing number of freqs and times
    num_samples = 3020 #samplerate/10
    f, t, Pxx = signal.spectrogram(samples,samplerate,signal.get_window('hamming',int(num_samples)))
    frequency_index = np.argmin(np.abs(f-freq))
    amps = Pxx[frequency_index,:]
    
    # smooth to remove vibrato
    # using gaussian kernel with sigma=2
    if smooth:
        kern_domain = np.linspace(-2, 2, 21+1)
        gaussian_kernel = st.norm.pdf(kern_domain)
        amps = np.convolve(amps, gaussian_kernel, mode='same')
    return t, amps

def plot_amp(freq, samplerate, samples, smooth=True):
    plt.ylabel('Power [db]')
    plt.xlabel('Time [sec]')
    plt.title(str(freq) + ' Hz')
    plt.plot(*get_freq(freq, samplerate, samples, smooth))


def plot_amps(fund_freq, samplerate, samples, smooth=True):
    npartials = 30; nrows = npartials//5; ncols = npartials//6
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*8, nrows*8))
    freq = fund_freq
    for i in range(nrows):
        for j in range(ncols):
            ax = axs[i,j]
            ax.set_ylabel('Power')
            ax.set_xlabel('Time [sec]')
            ax.set_title('Partial no: ' + str(i*ncols+j+1) + ' (' + str(freq) + ' Hz' + ')')
            ax.plot(*get_freq(freq, samplerate, samples, smooth))
            freq += fund_freq

    # adjust spacing
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.7,
                        hspace=1.0)
    plt.show()

def get_envs(fund_freq, samplerate, samples, smooth=True):
    """
    prints out a syntactic supercollider array for specifying
    the levels in an envelope
    """
    freq = fund_freq
    print('[', end = '')
    for i in range(30):
        t, amps = get_freq(freq, samplerate, samples, smooth)
        amps = amps/np.max(amps) # normalize to 0 to 1
        print('[', end = '')
        for amp in amps:
            print(str(round(amp, 5)) + ', ', end='')
        print('], ',end='')
        freq += fund_freq
    print(']')



#samplerate, samples = wavfile.read('./string_vln.b3.wav')
#b3 = 247
#plot_amps(b3, samplerate, samples)
#get_envs(b3, samplerate, samples, smooth=False)

samplerate, samples = wavfile.read('./string_vln.g3.wav')
g3 = 196
plot_amps(g3, samplerate, samples, smooth=True)
get_envs(g3, samplerate, samples, smooth=True)
