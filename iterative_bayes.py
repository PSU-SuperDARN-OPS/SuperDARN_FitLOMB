# jon klein, jtklein@alaska.edu
# 02/14

# equations to use bayesian analysis to determine the spectral content of a FitACF return
# references:
# (1)
# (2) 
# (3) "Bayesian Analysis. III. Applications to NMR Signal Detection, Model Selection and Parameter Estimation" by  G. Larry Bretthorst, 1990

# approach:
# 

import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
import scipy.signal as signal
import pdb
# look into numexpr
from scipy.optimize import curve_fit

# copied from jef spaleta's code..
def find_fwhm(ia, pt_apex,da=1):
      fwhm=0.0
      if pt_apex > 0 and pt_apex < ia.size-1:
        right_slice = ia[pt_apex:len(ia)-1:1]
        searcher=ia[pt_apex]
        i=0
        for i in xrange(len(right_slice)):
          if right_slice[i] < float(searcher)/2.:
            break
        i=max(1,i)
        left_slice = ia[pt_apex:0:-1]
        searcher=ia[pt_apex]
        j=0
        for j in xrange(len(left_slice)):
          if left_slice[j] < float(searcher)/2.:
            break
        j=max(1,j)
        fwhm=(i+j)*da
      if pt_apex == 0:
        right_slice = ia[pt_apex:len(ia)-1:1]
        searcher=ia[pt_apex]
        i=0
        for i in xrange(len(right_slice)):
          if right_slice[i] < float(searcher)/2.:
            break
        i=max(1,i)
        fwhm=(2*i)*da
      if pt_apex == ia.size-1:
        left_slice = ia[pt_apex:0:-1]
        searcher=ia[pt_apex]
        j=0
        for j in xrange(len(left_slice)):
          if left_slice[j] < float(searcher)/2.:
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
def iterative_bayes(samples, t, freqs, alfs, maxfreqs = 4, env_model = 1):
    fits = []
    for i in range(maxfreqs):
        fit = calculate_bayes(samples, t, freqs, alfs, env_model = 1)
        fits.append(fit)
        samples -= fit['amplitude'] * np.exp(1j * 2 * np.pi * t * fit['frequency']) * np.exp(-t * fit['alpha'])
    return fits

def calculate_bayes(s, t, f, alfs, env_model = 1):
    N = len(t) * 2# see equation (10) in [4]
    m = 2
    dbar2 = (sum(np.real(s) ** 2) + sum(np.imag(s) ** 2)) / (N) # (11) in [4] 

    omegas = 2 * np.pi * f

    # create ce_matrix and se_matrix..
    # cos(w * t) * exp(-alf * t) for varying frequency, decay, and time 
    c_matrix = np.cos(np.outer(omegas, t))
    s_matrix = np.sin(np.outer(omegas, t))
    
    ce_matrix = np.zeros([len(omegas), len(t), len(alfs)])
    se_matrix = np.zeros([len(omegas), len(t), len(alfs)])
    
    # about 80% of execution time spent here..
    # ~ 400000
    # kernprof.py -l iterative_bayes.py
    # python -m line_profiler iterative_bayes.py.lprof

    # so... 
    envelope = np.exp(np.outer(-(alfs ** env_model), t))

    for (k, alf) in enumerate(alfs):
        for (j, ti) in enumerate(t):
            ce_matrix[:,j,k] = c_matrix[:,j] * envelope[k, j] 
            se_matrix[:,j,k] = s_matrix[:,j] * envelope[k, j] 

    # create R_f and C_f (12) and (13) in [4]
    # these reduce to the real and complex parts of the fourier transform for uniformly sampled data
    # matricies, len(freqs) by len(samples)
    # omegas * times * alphas
    # TODO: [12] has real - imag, but jef has real + imag. only jef's way works.. why?
    # about 10% of execution time spent here
    R_f = (np.dot(np.real(s), ce_matrix) + np.dot(np.imag(s), se_matrix)).T
    I_f = (np.dot(np.real(s), se_matrix) - np.dot(np.real(s), ce_matrix)).T
    
    
    # calculate I_f and S_f (14) and (15) in [4]
    # simultaneous sampling, so C_f and S_f reduce to the total power in envelope model
    z_matrix = np.ones([len(omegas), len(alfs)])
    for (k, alf) in enumerate(alfs): 
        z_matrix[:,k] *= sum(envelope[k,:] ** 2)

    C_f = z_matrix.T
    S_f = z_matrix.T

    # should S_r * C_i be removed for this case?
    # hbar2 is a "sufficient statistic" 
    hbar2 = ((R_f ** 2) / C_f + (I_f ** 2) / S_f) / 2.# (19) in [4] 
    
    P_f = ((N * dbar2 - hbar2) ** ((2 - N) / 2.)) / np.sqrt(C_f * S_f) # (18) in [4]
    
    P_f /= P_f.sum() # this is probably not valid..
    
    #P_f = ne.evaluate('((N * dbar2 - hbar2) ** ((2 - N) / 2.)) / sqrt(C_f * S_f)')
    # see "Nonuniform Sampling: Bandwidth and Aliasing"
    # for <sigma**2> and P(f|DI)

    #sigma2 = (N * dbar2 - hbar2) / (N - 4.)
    #prob_f = np.exp(hbar2 / (2. * sigma2))


    maxidx = np.argmax(P_f)
    max_tuple = np.unravel_index(maxidx, P_f.shape)

    alf_slice = P_f[max_tuple[0],:]
    omega_slice = P_f[:,max_tuple[1]]

    alf_fwhm = find_fwhm(alf_slice, max_tuple[1])
    omega_fwhm = find_fwhm(omega_slice, max_tuple[0])

    fit = {}
    fit['amplitude'] = R_f[max_tuple] / C_f[max_tuple]
    fit['frequency'] = f[max_tuple[1]]
    fit['frequency_fwhm'] = omega_fwhm
    fit['alpha'] = alfs[max_tuple[0]]
    fit['alpha_fwhm'] = alf_fwhm

    plt.plot(np.real(s))
    plt.plot(np.imag(s))
    fitsignal = fit['amplitude'] * np.exp(1j * 2 * np.pi * fit['frequency'] * t) * np.exp(-fit['alpha'] * t)
    plt.plot(np.real(fitsignal))
    plt.plot(np.imag(fitsignal))
    plt.show()        

    print 'alf: ' + str(fit['alpha'])
    print 'alf_fwhm: ' + str(fit['alpha_fwhm'])
    print 'frequency: ' + str(fit['frequency'])
    print 'frequency_fwhm: ' + str(fit['frequency_fwhm'])
    print 'amplitude: ' + str(fit['amplitude'])
         
    return fit 

    # calculate amplitude estimate from A_est[max] = R_est[max] / C_est[max]
    # B_est from  I_est[max] / S_est[max] (what is it?)
    # SNR from ...

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
    
