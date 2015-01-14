import numpy as np
import pdb

OVERLAP_THRESH = .20

# TODO: handle case with lag 0 - if lag 0 is bad, use alternate lag zero (if it exists..)
# (regenerate behavior for rawacf...)

# mask out lags-ranges at transmit pulses
#@profile
def good_lags_txsamples(prm):
  # Transmit samples
  ptimes_usec=[]
  psamples=[]
  smpoff=prm.lagfr/prm.smsep
  good_lags=[]

  for pulse in prm.ptab:
       t1 = pulse * prm.mpinc-prm.txpl/2.
       t2 = t1 + 3 * prm.txpl/2. + 100 
       ptimes_usec.append([t1,t2])
       psamples.append([int(t1/prm.smsep),int(t2/prm.smsep)])

  for rbin in xrange(prm.nrang): 
       lag_state=[]
       for l in xrange(prm.mplgs): # TODO: fix this, use alternate lag 0 if first is bad
            lag = prm.ltab[l]
            sam1 = lag[0]*(prm.mpinc/prm.smsep) + rbin +smpoff
            sam2 = lag[1]*(prm.mpinc/prm.smsep) + rbin +smpoff
            

            good=True
            for smrange in psamples:
              if (sam1 >= smrange[0]) and (sam1 <= smrange[1]): 
                  good=False 
              if (sam2 >= smrange[0]) and (sam2 <= smrange[1]): 
                  good=False
            lag_state.append(good)
       good_lags.append(lag_state)
   
  return np.array(good_lags,dtype="bool")


# mask out lag-ranges that overlap with other pulses
#@profile
def good_lags_overlap(prm, nolagzero = False):
  good_lags = np.zeros([prm.nrang, prm.mplgs])
  sampfr = prm.lagfr / prm.smsep # range gate of first range in lag0

  # calculate number of range-samples in the pulse sequence 
  nsamples = prm.ptab[-1] * (prm.mpinc / prm.smsep) + prm.nrang + sampfr
  pulse_returns = []
  
  # convolve pulse impulses with pwr0 to estimate of pulse returns
  # this is inefficient, maybe just pad pulse sequence.. 
  for pulse in prm.ptab:
      pulse_impulse = np.zeros(nsamples)
      t_pulse = pulse * (prm.mpinc / prm.smsep) + sampfr
      pulse_impulse[t_pulse] = 1
      pulse_returns.append(np.convolve(pulse_impulse, prm.pwr0))
    
  combined_return = np.zeros(nsamples + prm.nrang - 1)
  
  # estimate combined power from all pulses using pwr0
  for pulse in pulse_returns:
      combined_return += pulse
  
  # create mask of good samples from each pulse return where 
  # sample power comes from that sample by a ratio of at least OVERLAP_THRESH 
  # shift per-pulse return masks to overlap match range gates
  for (i,pulse) in enumerate(pulse_returns):
      pulse = (pulse / combined_return) > OVERLAP_THRESH
      startidx = prm.ptab[i] * (prm.mpinc / prm.smsep) + sampfr
      endidx = startidx + prm.nrang
      pulse_returns[i] = pulse[startidx:endidx]

  # reformat lag table to index pulse return mask list
  tab_lookup = {}
  lag_pairs = []
  for (i,p) in enumerate(prm.ptab):
      tab_lookup[p] = i
  for pair in prm.ltab:
      lag_pairs.append([tab_lookup[pair[0]],tab_lookup[pair[1]]])

  # create bad lags mask using lag table and per-pulse range gate masks
  for (j,pair) in enumerate(lag_pairs):
      p1ranges = pulse_returns[pair[0]] 
      p2ranges = pulse_returns[pair[1]]
      good_lags[:,j] = p1ranges & p2ranges 
 
  if nolagzero:
      good_lags[:,0] = 0
  
  return np.array(good_lags, dtype="bool")

#@profile
def get_bad_lags(prm, txlags = None):
    if txlags == None:
        txlags = good_lags_txsamples(prm)
    overlap_lags = good_lags_overlap(prm) 
    return ~txlags | ~overlap_lags
