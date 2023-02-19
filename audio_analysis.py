#! /usr/bin/env python
import math
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fft
import numpy as np
import scipy.stats as st

def smooth_data(x):
    # smooth to remove vibrato in get_freq, or just remove noise
    # using gaussian kernel with sigma=2
    kern_domain = np.linspace(-2, 2, 21+1)
    gaussian_kernel = st.norm.pdf(kern_domain)
    return np.convolve(x, gaussian_kernel, mode='same')

def get_freq(freq, samplerate, samples, smooth=True):
    # good value empirically for maximizing number of freqs and times
    num_samples = 3020 #samplerate/10
    f, t, Pxx = signal.spectrogram(samples,samplerate,signal.get_window('hamming',int(num_samples)))
    frequency_index = np.argmin(np.abs(f-freq))
    amps = Pxx[frequency_index,:]

    if smooth:
        amps = smooth_data(amps)
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

def print_array(arr):
    print('[', end = '')
    for x in arr:
        print(str(round(x, 5)) + ', ', end='')
    print('], ',end='')

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
        print_array(amps)
        freq += fund_freq
    print(']')

def timbrel_change(smooth = True, print_out = False):
    # these values obtained by isolating a quiet/loud segment of a solo violin performance
    # running audacity's spectral analysis to get amplitudes for various frequencies
    # and taking the first 30 partials after finding the fundamental frequency
    loud_partial_amps = np.array([-51.93972 , -49.4104  , -46.828625, -56.960018, -64.312508,
       -62.737423, -29.59796 , -62.45232 , -64.916168, -64.447052,
       -70.183189, -72.147018, -65.468178, -70.745598, -59.947681,
       -59.926273, -69.282608, -74.263039, -59.553307, -63.807056,
       -94.537758, -62.298313, -78.946373, -74.343536, -64.548203,
       -82.339119, -84.327194, -70.587006, -74.463646, -83.49617 ])
    quiet_partial_amps = np.array([ -65.825394,  -34.847195,  -52.882778,  -49.364624,  -54.84634 ,
        -51.980816,  -59.094482,  -48.69043 ,  -66.581573,  -54.611156,
        -74.646492,  -59.795284,  -81.939926,  -80.814651,  -77.027634,
        -76.393929,  -89.03508 ,  -72.259262,  -72.2481  ,  -74.552094,
       -106.836113,  -81.866112,  -94.832108,  -79.970268,  -83.187508,
        -89.181671,  -98.206154,  -94.213165,  -98.791794, -100.895943])

    if smooth:
        loud_partial_amps = smooth_data(loud_partial_amps)
        quiet_partial_amps = smooth_data(quiet_partial_amps)

    # plot data
    fig,axs = plt.subplots(3,1)
    axs[0].plot(loud_partial_amps)
    axs[0].set_title("Amplitude vs Partial No for Loud Violin")
    axs[1].plot(quiet_partial_amps)
    axs[1].set_title("Amplitude vs Partial No for Quiet Violin")
    diff = loud_partial_amps - quiet_partial_amps
    axs[2].plot(diff)
    axs[2].set_title("Difference between Loud and Quiet Violin Partials")
    plt.show()

    if print_out:
        print_array(diff)




if __name__ == '__main__':
    # uncomment these lines to plot/get envelope data for the b3 violin file
    samplerate, samples = wavfile.read('./string_vln.b3.wav')
    b3 = 247
    plot_amps(b3, samplerate, samples, smooth=False)
    get_envs(b3, samplerate, samples)

    # uncomment these lines to plot/get envelope data for the g3 violin file
    #samplerate, samples = wavfile.read('./string_vln.g3.wav')
    #g3 = 196
    #plot_amps(g3, samplerate, samples, smooth=False)
    #get_envs(g3, samplerate, samples, smooth=True)

    # uncomment these lines to plot timbrel change for two amplitudes
    #timbrel_change(smooth=False)
    #timbrel_change(print_out=True)
