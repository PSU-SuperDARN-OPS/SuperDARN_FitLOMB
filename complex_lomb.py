# using equations from http://www.ann-geophys.net/29/2101/2011/angeo-29-2101-2011.pdf
# [1] ( A study of Traveling Ionospheric Disturbances and Atmospheric Gravity Waves using EISCAT Svalbard Radar IPY-data )

# jon klein, jtklein@alaska.edu
# 02/06/14

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pdb

from scipy.optimize import curve_fit

# calculate lomb peridogram for a complex signal with missing samples
# samples at times t, for freqs 
# len(freqs) should be at least 4 * len(t)
def lomb(samples, t, freqs, normalized = True):
    # check that freqs > 4N 
    # check t is positive..
    # check stuff and complain
    return np.absolute(np.array([_p_w(samples, t, f * 2 * np.pi, normalized) for f in freqs]))

# calculate significance of a power
# equation 3 in [1]
# TODO: this doesn't work well with multiple returns...
# shared power demotes everybody.. work out significance from p_0, or some other noise measure?
# instead.. calculate probability of a awgn with p_0 average power being above power z...
# also, this is invalid for small number of samples? see M. Zechmeister, M. Kurster 2009
def lomb_ztop(z, freqs):
    return 1-((1 - np.exp(-z)) ** len(freqs))



# calculates the probability of a point on the spectrum reaching the calculated power
# 
def lomb_significance(lo, freqs, p0):
    pass

# helper function for lomb peridogram
# where t is numpy array of times, w is frequency in radians
# equation 3 in [1]
def _p_w(samples, t, w, normalized = True):
    mean = np.mean(samples)

    if normalized:
        varscale = 1 / (2. * np.var(samples))

    else:
        varscale = 1.

    tau = sum(np.sin(2. * w * t)) / sum(np.cos(2. * w * t)) 
 
    n1 = sum((samples - mean) * np.cos(w * (t - tau)))
    n2 = sum((samples - mean) * np.sin(w * (t - tau)))
    d1 = np.sqrt(sum(np.cos(w * (t - tau)) ** 2))
    d2 = np.sqrt(sum(np.sin(w * (t - tau)) ** 2))
    return varscale * ((abs(n1 / d1) ** 2) + (abs(n2 / d2) ** 2))

# see SuperDARN Workshop, 26-31/05/2013, velocity error tutorial.. W / sqrt(Nlag) for now
# calculate information about a peak
# - power at lag zero for gaussian and lorentzian
# - spectral width

# the frequency response of a decaying exponential in the time domain
# p * exp(-a * |x|)
def lorentzian(freqs, a, p, df):
    return p * ((2 * a)/(a ** 2 + 4 * (np.pi ** 2) * ((freqs-df) ** 2)))

# the frequency reponse of a gaussian in the time domain
# p * exp(-alpha * (x ** 2))
def gaussian(freqs, alpha, p, df):
    return p * np.sqrt(np.pi / alpha) * np.exp(-(((np.pi * (freqs-df))**2)/alpha))

# lo - numpy array with spectrum
# prob - significance array
# freqs - numpy array of frequencies 

# returns an array of indexes of input array
# for items in array centered on centeridx which are True
# this is probably not a pythonic approach.. see image segmentation?
def get_segment(array, centeridx):
    if not array[centeridx] == True:
        raise ValueError('Specified center index of array in get_segment is False!')

    segmentidxs = [centeridx]

    _t = centeridx - 1
    while array[_t] and _t >= 0:
        segmentidxs = [_t] + segmentidxs
        _t -= 1
        

    _t = centeridx + 1
    while array[_t] and _t < len(array):
        segmentidxs = segmentidxs + [_t]
        _t += 1

    return np.array(segmentidxs)

if __name__ == '__main__':
    fs = 1 
    t_total = 25 * (1/fs)

    NOISE_SCALE = .05

    t = np.arange(0,t_total,1./fs)

    noise = np.random.normal(scale=NOISE_SCALE,size=len(t)) + 1j * np.random.normal(scale=NOISE_SCALE,size=len(t))
    sin1 = 1 * np.exp(1j * 2 * np.pi * t * .2) * np.exp(-t *.5) 
    sin2 = 1 * np.exp(1j * 2 * np.pi * t * .3) * np.exp(-t *.6) 
    sin3 = 1 * np.exp(1j * 2 * np.pi * t * .4) * np.exp(-t *.5) 
    sin4 = 1 * np.exp(1j * 2 * np.pi * t * .05) * np.exp(-t * .02)
    samples =  sin1 + sin4 + noise

    freqs = np.linspace(-fs/2.,fs/2.,len(t) * 4)
    lo = lomb(samples, t, freqs, normalized = True)
    prob = lomb_ztop(lo, freqs)
    lo *= ((2 * np.std(samples)) / len(freqs))# denormalize data after computing probabilities

    # plot results
    plt.subplot(3,1,1)
    plt.plot(t, np.real(samples))
    plt.plot(t, np.imag(samples))
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    plt.title('time domain samples')
    plt.legend(['I', 'Q'])
    plt.grid(True)
    
    plt.subplot(3,1,2)
    plt.plot(freqs, lo)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('power')
    plt.title('lomb periodogram')
    plt.grid(True)
    
    plt.subplot(3,1,3)
    plt.plot(freqs,prob) 
    plt.xlabel('frequency (Hz)')
    plt.ylabel('certainty')
    plt.title('lomb periodogram certainty')
    plt.grid(True)
 
    plt.show()
    pdb.set_trace()
