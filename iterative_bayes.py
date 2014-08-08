# jef spaleta
# jon klein, jtklein@alaska.edu

# equations to use bayesian analysis to determine the spectral content of a FitACF return
# references:
# "Bayesian Analysis. III. Applications to NMR Signal Detection, Model Selection and Parameter Estimation" by  G. Larry Bretthorst, 1990
import time
import_start=time.clock()
import numpy as np
import numexpr as ne
from timecube import TimeCube, make_spacecube

import_elapsed=time.clock()-import_start
VERBOSE = False 

# fwhm modified from jef spaleta's code..
# -added variable "half" factor (for working with log values)
# TODO: zoom out if we zoomed in too far
# TODO: flag if fwhm is entire range
def find_fwhm(ia, pt_apex,log = True,factor=.5,da=1):
  if log:
    factor = (ia[pt_apex] + np.log10(factor))/(ia[pt_apex])
  
  fwhm=0.0
  bounded = 2

  if pt_apex > 0 and pt_apex < ia.size-1:
    right_slice = ia[pt_apex:len(ia)-1:1]
    searcher=ia[pt_apex]
    i=0
    for i in xrange(len(right_slice)):
      if right_slice[i] < float(searcher)*factor:
        bounded -= 1
        break
    i=max(1,i)
    left_slice = ia[pt_apex:0:-1]
    searcher=ia[pt_apex]
    j=0
    for j in xrange(len(left_slice)):
      if left_slice[j] < float(searcher)*factor:
        bounded -= 1
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
  return fwhm, bounded

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
def iterative_bayes(samples, t, freqs, alfs, env_model, maxfreqs, cubecache = False, zoom = 10., zoomspan = 3.,max_iterations=10):
    fits = []
    cubecache = False
    if not cubecache:
        timecube = (make_spacecube(t, freqs, alfs, env_model))
    else:
        timecube = False
    for i in range(maxfreqs):
        # calculate initial fit
        fit = calculate_bayes(samples, t, freqs, alfs, env_model, cubecache = cubecache, timecube = timecube)
        zoom_iteration=0 
 
        if zoom:
            #print "zoom: enabled"
            # this may disrupt fwhm calculations
            # zoom frequency range in on fit, recalculate bayes for increased resolution
            # don't cache zoomed timecubes
            coarsefwhm_f = fit['frequency_fwhm']
            coarsefwhm_a = fit['alpha_fwhm']
            coarse_f_lims=fit['frequency_lims']
            coarse_a_lims=fit['alpha_lims']
            coarse_p=fit["p"].copy()

            f_zoom_needed=False
            a_zoom_needed=False
            f_delta=fit['frequency_lims'][1]-fit['frequency_lims'][0]
            a_delta=fit['alpha_lims'][1]-fit['alpha_lims'][0]
            if f_delta > 0.0:
              f_ratio=coarsefwhm_f/(f_delta)
              if f_ratio < 0.4 : f_zoom_needed=True
              else: f_zoom_needed=False
            else:  
                f_ratio=np.nan
                f_zoom_needed=False
            if a_delta > 0.0:
              a_ratio=coarsefwhm_a/(a_delta)
              if a_ratio < 0.4 : a_zoom_needed=True
              else: a_zoom_needed=False
            else:  
                a_ratio=np.nan
                a_zoom_needed=False

            #print "Coarse Ratio:",f_ratio,a_ratio
            #print "Coarse F_lims:",fit['frequency_lims']
            #print "Coarse F_fhwm:",fit['frequency_fwhm'],f_delta
            #print "Coarse A_lims:",fit['alpha_lims']
            #print "Coarse A_fhwm:",fit['alpha_fwhm'],a_delta
            lastfwhm_f=coarsefwhm_f 
            lastfwhm_a=coarsefwhm_a 
            zfreqs=freqs
            zalfs=alfs
            f_pixel=(zfreqs[-1]-zfreqs[0])/len(zfreqs)
            while (f_zoom_needed or a_zoom_needed) and (zoom_iteration < max_iterations):
                 #print "Zoom iteration:",zoom_iteration,zfreqs[0],zfreqs[-1],f_pixel
                 zoom_iteration+=1
                 if f_zoom_needed:
                   #print "  F Zoom Needed %f" % (lastfwhm_f)
                   zfreqs = calc_zoomvar(zfreqs, fit['frequency'], lastfwhm_f,zoomspan, zoom)
                 if a_zoom_needed:
                   #print "  A Zoom Needed"
                   zalfs = calc_zoomvar(zalfs, fit['alpha'], lastfwhm_a, zoomspan,zoom)
                 zoomcube = (make_spacecube(t, zfreqs, zalfs, env_model))
                 fit = calculate_bayes(samples, t, zfreqs, zalfs, env_model, cubecache = False, timecube = zoomcube)

                 # use old fwhm if fwhm was bounded by zoom level, revert to old fwhm (it wasn't a good fit anyways..)
                 if fit['frequency_fwhm_bounded']:
                     #fit['frequency_fwhm'] = lastfwhm_f
                     #print "warning: frequency_fwhm_bounded"
                     pass 
                 if fit['alpha_fwhm_bounded']:
                     #fit['alpha_fwhm'] = lastfwhm_a
                     #print "warning: alpha_fwhm_bounded"
                     pass
                 lastfwhm_f = fit['frequency_fwhm']
                 lastfwhm_a = fit['alpha_fwhm']
                 f_zoom_needed=False
                 a_zoom_needed=False
                 f_delta=fit['frequency_lims'][1]-fit['frequency_lims'][0]
                 a_delta=fit['alpha_lims'][1]-fit['alpha_lims'][0]
                 if f_delta > 0.0:
                   f_ratio=lastfwhm_f/(f_delta)
                   if f_ratio < 0.4 : f_zoom_needed=True
                   else: f_zoom_needed=False
                 else:  
                     f_ratio=np.nan
                     f_zoom_needed=False
                 if a_delta > 0.0:
                    a_ratio=lastfwhm_a/(a_delta)
                    if a_ratio < 0.4 : a_zoom_needed=True
                    else: a_zoom_needed=False
                 else:  
                     a_ratio=np.nan
                     a_zoom_needed=False
                 #print "  Zoom Ratio:",f_ratio,a_ratio
                 #print "    F_lims:",fit['frequency_lims']
                 #print "    F_fhwm:",fit['frequency_fwhm'],f_delta
                 #print "    A_lims:",fit['alpha_lims']
                 #print "    A_fhwm:",fit['alpha_fwhm'],a_delta
        fit['r2_phase'] = calc_phase_r2(fit['signal'], samples)
        samples -= fit['signal']
        fit['fit_snr'] = np.mean(abs(fit['signal']) ** 2) / np.mean(abs(samples) ** 2)
        fit["coarse_p"]=coarse_p.copy()
        fit["coarse_frequency_lims"]=coarse_f_lims
        fit["coarse_alpha_lims"]=coarse_a_lims
        fit["zoom_iteration"]=zoom_iteration
        fits.append(fit)
    return fits

# calculate r^2 error for phase fit
# fitsignal - fitted complex values for signal in lagspace at good lags
# samples - measured complex values of good lags
def calc_phase_r2(fitsignal, samples):
    # calculate unwrapped phase at lags
    p_meas = np.unwrap(np.arctan2(np.imag(samples), np.real(samples)))
    p_fit = np.unwrap(np.arctan2(np.imag(fitsignal), np.real(fitsignal)))

    # using notation from http://en.wikipedia.org/wiki/Coefficient_of_determination
    ss_tot = sum((p_meas - np.mean(p_meas)) ** 2.)
    ss_res = sum((p_meas - p_fit)**2.)
    r2 = 1. - (ss_res/ss_tot)
    return r2

# calculates zoomed parameters
def calc_zoomvar(ar, center, span,scale,length):
    znar = length 
    za0 = center - span*scale/2. 
    za1 = center + span*scale/2.
    return np.linspace(max(za0, ar[0]), min(za1, ar[-1]), znar)

# to profile:
# kernprof.py -l foo.py
# python -m line_profiler foo.py.lprof
#@profile
def calculate_bayes(s, t, f, alfs, env_model, cubecache = False, timecube = False):
    N = len(t) * 2.# see equation (10) in [4]
    m = 2

    dbar2 = (sum(np.real(s) ** 2) + sum(np.imag(s) ** 2)) / (N) # (11) in [4] 
    # reuse time x alphas x frequency cube if possible, this is time consuming to construct
    if cubecache:
        ce_matrix, se_matrix, CS_f = cubecache.get_spacecube(t, f, alfs, env_model) 
    elif timecube:
        ce_matrix, se_matrix, CS_f = timecube 
    else:
        ce_matrix, se_matrix, CS_f = make_spacecube(t, f, alfs, env_model)

    # create R_f and I_f (12) and (13) in [4]
    # these reduce to the real and complex parts of the fourier transform for uniformly sampled data
    # matricies, len(freqs) by len(samples)
    # omegas * times * alphas
    # TODO: [12] has real - imag, but jef has real + imag. only jef's way works.. why?
    # about 50% of execution time is spent here
    R_f = (np.dot(np.real(s), ce_matrix) + np.dot(np.imag(s), se_matrix)).T
    I_f = (np.dot(np.real(s), se_matrix) - np.dot(np.imag(s), ce_matrix)).T
    
    # we might be able to eliminate constants.. considering that we blow them away with the normalization anyways
    # hbar2 is a "sufficient statistic" 
    # execution time halved by using numexpr, about 5% of execution time is spent here
    hbar2 = ne.evaluate('((R_f ** 2) / CS_f + (I_f ** 2) / CS_f) / 2.')# (19) in [4] 
    
    # use logarithms to avoid underflow (** 20 will drown large probabilities..)
    # about 30% of execution time is spent here
    P_f = np.log10(N * dbar2 - hbar2)  * ((2 - N) / 2.) - np.log10(CS_f)
   
    # see "Nonuniform Sampling: Bandwidth and Aliasing"
    # for <sigma**2>
    # sigma2 = (N * dbar2 - hbar2) / (N - 4.) # ???
    maxidx = np.argmax(P_f)
    max_tuple = np.unravel_index(maxidx, P_f.shape)
    alf_slice = P_f[:,max_tuple[1]]
    freq_slice = P_f[max_tuple[0],:]

    alf_fwhm, a_fwhm_bounded = find_fwhm(alf_slice, max_tuple[0])
    freq_fwhm, f_fwhm_bounded = find_fwhm(freq_slice, max_tuple[1])
    # TODO: add comparison of input samples to std deviation of input samples 
   
    fit = {}
    fit['amplitude'] = (R_f[max_tuple] + I_f[max_tuple]) / CS_f[max_tuple] 
    fit['amplitude_error_unscaled'] = CS_f[max_tuple]
    fit['frequency'] = f[max_tuple[1]]
    if len(f) > 1 : fit['frequency_fwhm'] = freq_fwhm * abs(f[1] - f[0])
    else: fit['frequency_fwhm'] = np.nan 
    fit['frequency_fwhm_bounded'] = f_fwhm_bounded
    fit['alpha'] = alfs[max_tuple[0]] 
    if len(alfs) > 1 : fit['alpha_fwhm'] = alf_fwhm * abs(alfs[1] - alfs[0])
    else: fit['alpha_fwhm'] = np.nan 
    fit['alpha_fwhm_bounded'] = a_fwhm_bounded
    fit['samples'] = s.copy()
    fit['t'] = t.copy()
    fit['signal'] = fit['amplitude'] * np.exp(1j * 2 * np.pi * fit['frequency'] * t) * np.exp(-fit['alpha'] * t)
    fit['p'] = P_f.copy()
    if len(f) > 1: fit['frequency_lims'] = (min(f),max(f)) 
    else: fit['frequency_lims']=(min(f),min(f)+1)
    if len(alfs) > 1: fit['alpha_lims'] = (min(alfs),max(alfs)) 
    else: fit['alpha_lims']=(min(alfs),min(alfs)+1)
    return fit 

    # calculate amplitude estimate from A_est[max] = R_est[max] / C_est[max]

if __name__ == '__main__':
    print "Initial module import time: %f [sec]" % (time.clock()-import_start)
    plot=False
    if plot: 
      pyplot_start=time.clock()
      import matplotlib.pyplot as plt
      print "matplotlib pyplot import time: %f [sec]" % (time.clock()-pyplot_start)
    num_records=100
    fs = 100.
    ts = 1./fs

    f = [30.]
    amp = [10.]
    alf = [10.]

    lags = np.arange(0, 50) * ts
    times=[]
    signal=[]
    for i in xrange(num_records):
      F=f[0]+0.1*np.random.randn()+float(i-num_records/2)/float(num_records)*.1*f[0] 
      U=alf[0]+0.1*np.random.randn() 
      A=amp[0]+0.1*np.random.rand() 
      for t,lag in enumerate(lags): 
        times.append(lag)
        N_I=0.1*np.random.randn()
        N_Q=0.1*np.random.randn()
        N_A=np.sqrt(N_I**2+N_Q**2)
        N_phase=np.tan(N_Q/N_I)
        sig=A * np.exp(1j * 2 * np.pi * F * lag) * np.exp(-U * lag)+N_A*np.exp(1j*N_phase)
        signal.append(sig)
    #sin1 = amp[0] * np.exp(1j * 2 * np.pi * f[0] * t) * np.exp(-alf[0] * t)
    #sin2 = amp[1] * np.exp(1j * 2 * np.pi * f[1] * t) * np.exp(-alf[1] * t)
    #samples = sin1+sin2 
    samples=np.array(signal)
    times=np.array(times)
    if plot:
      #plt.figure(10)
      #plt.plot(np.real(sin1),label="Real 1")
      #plt.plot(np.imag(sin1),label="Imag 1")
      #plt.figure(20)
      #plt.plot(np.real(sin2),label="Real 2")
      #plt.plot(np.imag(sin2),label="Imag 2")
      plt.figure(30)
      plt.scatter(times,np.real(samples),label="Real",color="blue")
      plt.scatter(times,np.imag(samples),label="Imag",color="green")

    freqs = np.linspace(-fs/2, fs/2, 50)
    alfs = np.linspace(0,fs/2., 20)
    start=time.clock()
    fits = iterative_bayes(samples, times, freqs, alfs, 1, 1)
    print "Time Iterative Bayes: %f [sec]" % (time.clock()-start)
    for i,fit in enumerate(fits):
        print 'amp: %f %f' % (amp[i],fit['amplitude'])
        print 'freq: %f %f' % (f[i],fit['frequency'])
        print 'freq_fwhm: ' + str(fit['frequency_fwhm'])
        print 'freq_lims: ', fit['frequency_lims']
        print 'alf: %f %f' % (alf[i],fit['alpha'])
        print 'alf_fwhm: ' + str(fit['alpha_fwhm'])
        print 'alpha_lims: ', fit['alpha_lims']
        print 'coarse_p:', fit['coarse_p'].shape
        print 'zoom iteration:' + str(fit['zoom_iteration'])
        print 'p:', fit['p'].shape
        model=fit['amplitude']*np.exp(1j * 2 * np.pi * fit['frequency'] * lags) * np.exp(-fit['alpha'] * lags)
        if i==0 :
          full_model=model
        else : 
          full_model=full_model+model
        if plot:
          #plt.figure(10*(i+1))
          #plt.plot(np.real(model),label="Fit %d Real" % (i+1))
          #plt.plot(np.imag(model),label="Fit %d Imag" % (i+1))
          plt.figure(10*(i+1)+1)
          plt.imshow(fit['coarse_p'],origin="lower",aspect='auto',
            extent=(fit['coarse_frequency_lims'][0],fit['coarse_frequency_lims'][1],fit['coarse_alpha_lims'][0],fit['coarse_alpha_lims'][1])
          )
          plt.figure(10*(i+1)+2)
          plt.imshow(fit['p'],origin="lower",aspect='auto',
            extent=(fit['frequency_lims'][0],fit['frequency_lims'][1],fit['alpha_lims'][0],fit['alpha_lims'][1])
          )
    if plot:
      #plt.figure(10)
      #plt.legend()
      #plt.figure(20)
      #plt.legend()
      plt.figure(30)
      plt.plot(lags,np.real(full_model),label="Fit Real")
      plt.plot(lags,np.imag(full_model),label="Fit Imag")
      plt.legend()

      plt.show()
