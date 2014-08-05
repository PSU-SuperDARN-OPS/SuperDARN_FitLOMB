import numpy as np
import pdb
# TODO: handle case with lag 0 - if lag 0 is bad, use alternate lag zero (if it exists..)
# (regenerate behavior for rawacf...)
def good_lags(prm,pwr0):

# Transmit samples
  ptimes_usec=[]
  psamples=[]
  smpoff=prm.lagfr/prm.smsep
  for pulse in prm.ptab:
       t1 = pulse * prm.mpinc-prm.txpl/2.
       t2 = t1 + 3 * prm.txpl/2. + 100 
       ptimes_usec.append([t1,t2])
       psamples.append([int(t1/prm.smsep),int(t2/prm.smsep)])

  good_lags=[]
  for rbin in xrange(prm.nrang): 
       lag_state=[]
       for l in xrange(prm.mplgs): # TODO: fix this, use alternate lag 0 if first is bad
            lag = prm.ltab[l]
            sam1 = lag[0]*(prm.mpinc/prm.smsep) + rbin +smpoff
            sam2 = lag[1]*(prm.mpinc/prm.smsep) + rbin +smpoff
            
            good=True
            for smrange in psamples:
              if (sam1 >= smrange[0]) and (sam1 <= smrange[1]): good=False 
              if (sam2 >= smrange[0]) and (sam2 <= smrange[1]): good=False
            lag_state.append(good)
       good_lags.append(lag_state)
# Range overlap
  range_overlap=[]
  for ck_pulse in xrange(prm.mppul):
    overlap=[]
    for pulse in xrange(prm.mppul):
      diff_pulse = prm.ptab[ck_pulse] - prm.ptab[pulse]
      overlap.append(diff_pulse * prm.mpinc/prm.smsep)
    range_overlap.append(overlap)

  for rbin in xrange(prm.nrang):
    bad_pulse=[]
    for pulse in xrange(prm.mppul):
      bad_pulse.append(0);
    for ck_pulse in xrange(prm.mppul):
      for pulse in xrange(prm.mppul):
        ck_range = range_overlap[ck_pulse][pulse] + rbin;
        if ((pulse != ck_pulse) and (0 <= ck_range) and (ck_range < prm.nrang)):
          pwr_ratio = 1;  #pwr_ratio = (long) (nave * MIN_PWR_RATIO);
          min_pwr =  pwr_ratio * pwr0[rbin];
          if(min_pwr < pwr0[ck_range]):
            bad_pulse[ck_pulse]==1  
  # mark the bad lags 
    for pulse in xrange(prm.mppul):
      if (bad_pulse[pulse] == 1): 
        for lag in xrange(prm.mplgs):
          for i in xrange(2):
            if (prm.ltab[lag][i] == prm.ptab[pulse]):
              good_lags[rbin][lag]=False;  
              pass
  return np.array(good_lags,dtype="bool")
 
def bad_lags(prm,pwr0):
  return ~good_lags(prm,pwr0)
