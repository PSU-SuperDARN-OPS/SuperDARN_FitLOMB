# jon klein, jtklein@alaska.edu
# 02/14

# equations to use bayesian analysis to determine the spectral content of a FitACF return
# references:
# (1)
# (2) 
# (3) "Bayesian Analysis. III. Applications to NMR Signal Detection, Model Selection and Parameter Estimation" by  G. Larry Bretthorst, 1990

# approach:
# 

# TODO: add signal to noise ratio for calculations
# TODO: add lambda power
# TODO: add fit certainty

import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
import pdb
# look into numexpr
from timecube import TimeCube

VERBOSE = False 
VEL_CMAP = plt.cm.PuOr
FREQ_CMAP = plt.cm.spectral
NOISE_CMAP = plt.cm.autumn
SPECW_CMAP = plt.cm.hsv
POWER_CMAP = plt.cm.jet

# jef spaleta's code..
# modified variable "half" factor (for working with logs)
# may be better to fit a spline
# (see http://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak)
# however, this would assume that a spline is a good model, that approach may break with multiple peaks
# I'm not convinced that this approach is valid for peaks near edges (near DC or high frequencies, or high or low decay rates)

def find_fwhm(ia, pt_apex,log = True,factor=.5,da=1):
  if log:
    factor = (ia[pt_apex] + np.log10(factor))/(ia[pt_apex])
  
  fwhm=0.0

  if pt_apex > 0 and pt_apex < ia.size-1:
    right_slice = ia[pt_apex:len(ia)-1:1]
    searcher=ia[pt_apex]
    i=0
    for i in xrange(len(right_slice)):
      if right_slice[i] < float(searcher)*factor:
        break
    i=max(1,i)
    left_slice = ia[pt_apex:0:-1]
    searcher=ia[pt_apex]
    j=0
    for j in xrange(len(left_slice)):
      if left_slice[j] < float(searcher)*factor:
        break
    j=max(1,j)
    fwhm=(i+j)*da
  if pt_apex == 0:
    right_slice = ia[pt_apex:len(ia)-1:1]
    searcher=ia[pt_apex]
    i=0
    for i in xrange(len(right_slice)):
      if right_slice[i] < float(searcher)*factor:
        break
    i=max(1,i)
    fwhm=(2*i)*da
  if pt_apex == ia.size-1:
    left_slice = ia[pt_apex:0:-1]
    searcher=ia[pt_apex]
    j=0
    for j in xrange(len(left_slice)):
      if left_slice[j] < float(searcher)*factor:
        break
    j=max(1,j)
    fwhm=(2*j)*da

  return fwhm

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

# calcuates the peak frequency component using bayesian analysis given complex samples, an envelope model, and sample times
# returns model parameters?
# s - samples (complex numbers)
# t - sample times (may not be continuous)
# ts - sampling rate 
# alfs - range of time constants in exponential to check 
# env_model - 1 for labmda and 2 for sigma fit

# returns fit, with model parameters, a frequency, and a significance
# for simultaneous complex samples, normalized frequency, ts normalized to 1s
#@profile
def iterative_bayes(samples, t, freqs, alfs, cubecache, maxfreqs = 4, fmax = 4e3, env_model = 1):
    fits = []
    for i in range(maxfreqs):
        fit = calculate_bayes(samples, t, freqs, alfs, cubecache, env_model = 1)
        fits.append(fit)
        samples -= fit['signal']

    return fits

# to profile:
# kernprof.py -l foo.py
# python -m line_profiler foo.py.lprof
#@profile
def calculate_bayes(s, t, f, alfs, cubecache, env_model = 1):
    N = len(t) * 2# see equation (10) in [4]
    m = 2
    dbar2 = (sum(np.real(s) ** 2) + sum(np.imag(s) ** 2)) / (N) # (11) in [4] 
    
    ce_matrix, se_matrix, CS_f = cubecache.get_spacecube(t, f, alfs, env_model)

    # create R_f and I_f (12) and (13) in [4]
    # these reduce to the real and complex parts of the fourier transform for uniformly sampled data
    # matricies, len(freqs) by len(samples)
    # omegas * times * alphas
    # TODO: [12] has real - imag, but jef has real + imag. only jef's way works.. why?
    # about 50% of execution time is spent here
    R_f = (np.dot(np.real(s), ce_matrix) + np.dot(np.imag(s), se_matrix)).T
    I_f = (np.dot(np.real(s), se_matrix) - np.dot(np.real(s), ce_matrix)).T
    
    # we might be able to eliminate constants.. considering that we blow them away with the normalization anyways
    # hbar2 is a "sufficient statistic" 
    # execution time halved by using numexpr, about 5% of execution time is spent here
    hbar2 = ne.evaluate('((R_f ** 2) / CS_f + (I_f ** 2) / CS_f) / 2.')# (19) in [4] 
    
    # use logarithms to avoid underflow (** 20 will drown large probabilities..)
    # about 30% of execution time is spent here
    P_f = np.log10(N * dbar2 - hbar2)  * ((2 - N) / 2) - np.log10(CS_f)
#    P_f = ne.evaluate('log10((N * dbar2 - hbar2) ** ((2 - N) / 2.)) - log10(CS_f)') # (it takes a little longer to use numexpr..)

    # don't bother de-logging, we don't use this anyways.
    #P_f = 10 * pow(10., P_f)
    #P_f = ((N * dbar2 - hbar2) ** ((2 - N) / 2.)) / C_f # (18) in [4]
    #P_f /= P_f.sum() # this is probably not valid..
    
    # see "Nonuniform Sampling: Bandwidth and Aliasing"
    # for <sigma**2>
    #sigma2 = (N * dbar2 - hbar2) / (N - 4.) # ???

    maxidx = np.argmax(P_f)
    max_tuple = np.unravel_index(maxidx, P_f.shape)

    alf_slice = P_f[max_tuple[0],:]
    freq_slice = P_f[:,max_tuple[1]]

    alf_fwhm = find_fwhm(alf_slice, max_tuple[1])
    freq_fwhm = find_fwhm(freq_slice, max_tuple[0])

    fit = {}
    fit['amplitude'] = R_f[max_tuple] / CS_f[max_tuple]
    fit['frequency'] = f[max_tuple[1]]
    fit['frequency_fwhm'] = freq_fwhm 
    fit['alpha'] = alfs[max_tuple[0]]
    fit['alpha_fwhm'] = alf_fwhm
    fit['samples'] = s.copy()
    fit['t'] = t.copy()
    fit['signal'] = fit['amplitude'] * np.exp(1j * 2 * np.pi * fit['frequency'] * t) * np.exp(-fit['alpha'] * t)
    
    if(VERBOSE):
        print 'alf: ' + str(fit['alpha'])
        print 'alf_fwhm: ' + str(fit['alpha_fwhm'])
        print 'frequency: ' + str(fit['frequency'])
        print 'frequency_fwhm: ' + str(fit['frequency_fwhm'])
        print 'amplitude: ' + str(fit['amplitude'])

    if abs(fit['amplitude']) < 1e-9:
        print 'something went wrong with the fit.. dropping into debug mode'
        #pdb.set_trace()

    return fit 

    # calculate amplitude estimate from A_est[max] = R_est[max] / C_est[max]
    # B_est from  I_est[max] / S_est[max] (what is it?)

if __name__ == '__main__':
    fs = 1. 
    ts = 1./fs
    t_total = 20 * (1/fs)
    nfreqs_mult = 10
    NOISE_SCALE = .5 
    MAX_SIGNALS = 5
    alfs = np.linspace(0,1,50) # range of possible decay rates
    t = np.arange(0,t_total,1./fs)

    nfreqs = nfreqs_mult * len(t) 
    freqs = np.linspace(-.5/ts, .5/ts, nfreqs)

    noise = np.random.normal(scale=NOISE_SCALE,size=len(t)) + 1j * np.random.normal(scale=NOISE_SCALE,size=len(t))
    sin1 = 1 * np.exp(1j * 2 * np.pi * t * -.25) * np.exp(-t *.10) 
    sin2 = 2 * np.exp(1j * 2 * np.pi * t * .07) * np.exp(-t * .02)

    samples =  sin1 + sin2 + noise

    for si in range(MAX_SIGNALS):
        fit = calculate_bayes(samples, t, freqs, alfs)
        fit['index'] = si
        samples -= fit['amplitude'] * np.exp(1j * 2 * np.pi * t * fit['frequency']) * np.exp(-t * fit['alpha'])
    
    plt.show()        
