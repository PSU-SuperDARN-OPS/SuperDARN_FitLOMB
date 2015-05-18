import numpy as np
import pdb

OVERLAP_THRESH = .25

# TODO: handle case with lag 0 - if lag 0 is bad, use alternate lag zero (if it exists..)
# (regenerate behavior for rawacf...)

# mask out lags-ranges at transmit pulses
#@profile
def convo_good_lags_txsamples(prm):
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
def convo_good_lags_overlap(prm, nolagzero = False):
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

# code from jef to simulate fitacf badlags
def fitacf_more_badlags(w,good_lags,prm,noise_lev=0.0):
  badflag_1 = 0;
  badflag_2 = 0;
  # self correlated lag0 noise
  fluct0 =  w[0]/np.sqrt(2.0*prm.nave);
  # allowable lag power contaminated with noise
  fluct =  w[0] + 2.0*noise_lev+fluct0;
  fluct_old = fluct;
  k_old = 0;
  for k in xrange(prm.mplgs):
    good_lags[k]=True
    if (badflag_2):
      good_lags[k]=False
    else:
      # JDS: test to see if the lag pwr is too small compared to lag0
      #   I think this is bunk...but leave it in for fitacf comparison for now
      if (w[k] <= w[0]/np.sqrt(prm.nave)) :
        #print "too small: ",k,w[k],w[0]/np.sqrt(prm.nave)
        good_lags[k] = False
        # JDS: Disabling the latching badflag condition
        #  if you get 2 or 3 small pwr lags in a row badflag_2 becomes nonzero and latches, making the rests of the lags bad. 
        #  this really screws up in long sequence with large gaps. 
        #  disabling even though fitacf uses it
        #badflag_2 = badflag_1
        #badflag_1 = 1
      else:
        # JDS: lag pwr not too small compared to lag0 lets check against noise level
        badflag_1 = 0
        if (w[k] > fluct):
          #JDS: lag power larger than reasonable considering noise_lev input
          good_lags[k] = False
          # now lets do some pairwise comparison of power levels to make sure this lag is actually bad

          if (k < (prm.mplgs - 1)):
            # if  lag pwr is smaller than previous allowed max, ie current lag looks reasonable compared to last
            # and next lag is larger than current allowed max, ie next lag looks unreasonable compared to current
            # and next lag is smaller than current lag power...
            # mark the previous lag as bad and this one as good!
            if ((w[k] < fluct_old) and (w[k+1] > fluct) and (w[k+1] < w[k])):
              good_lags[k_old] = False
              good_lags[k] = True
        fluct_old = fluct
        fluct = 2.0*noise_lev + w[k] + fluct0
    if (good_lags[k]):
      k_old = k

  return good_lags

# code from jef to simulate fitacf badlags
def fitacf_good_lags(prm,pwr0,acfd):
  print "Acfd:",acfd.shape
  good_lags_tx=np.ones((prm.nrang,prm.mplgs),dtype='bool')
  good_lags_range=np.ones((prm.nrang,prm.mplgs),dtype='bool')
  good_lags_fluct=np.ones((prm.nrang,prm.mplgs),dtype='bool')
# Transmit samples
  ptimes_usec=[]
  psamples=[]
  psamples_offsetted=[]
  smpoff=prm.lagfr/prm.smsep
  for pulse in prm.ptab:
       t1 =pulse *prm.mpinc-prm.txpl/2.
       t2 = t1 + 3*prm.txpl/2. + 100
       ptimes_usec.append([t1,t2])
       psamples.append([int(t1/prm.smsep),int(t2/prm.smsep)])
       psamples_offsetted.append([int(t1/prm.smsep)+smpoff,int(t2/prm.smsep)+smpoff])
  print "Psamples:",psamples_offsetted
  for rbin in xrange(prm.nrang):
       lag_state=[]
       for l in xrange(prm.mplgs):
            lag=prm.ltab[l]
            sam1 = lag[0]*(prm.mpinc/prm.smsep) + rbin +smpoff
            sam2 = lag[1]*(prm.mpinc/prm.smsep) + rbin +smpoff

            good=True
            for smrange in psamples:
              if (sam1 >= smrange[0]) and (sam1 <= smrange[1]):
                good=False
              if (sam2 >= smrange[0]) and (sam2 <= smrange[1]):
                good=False
              if not good:
                #print "Bad Tx Lag:",rbin,l,lag
                pass
            good_lags_tx[rbin,l]=good
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
        ck_range = range_overlap[ck_pulse][pulse] + rbin
        if ((pulse != ck_pulse) and (0 <= ck_range) and (ck_range < prm.nrang)):
          pwr_ratio = 1;  #pwr_ratio = (long) (nave * MIN_PWR_RATIO);
          min_pwr =  pwr_ratio * pwr0[rbin];
          if(min_pwr < pwr0[ck_range]):
            #print "  :",prm.ptab[pulse],prm.ptab[ck_pulse],np.abs(prm.ptab[pulse]-prm.ptab[ck_pulse]) 
            bad_pulse[ck_pulse]=1
  # mark the bad lags 
    #print rbin,bad_pulse
    for pulse in xrange(prm.mppul):
      if (bad_pulse[pulse] == 1):
        for lag in xrange(prm.mplgs):
          #print "Range overlap pulse:",rbin,pulse,prm.ptab[pulse] 
          for i in xrange(2):
            if (prm.ltab[lag][i] == prm.ptab[pulse]):
              #print "Range overlap lag:",rbin,lag,prm.ltab[lag] 
              good_lags_range[rbin,lag]=False
              pass
    w=np.abs(acfd[rbin,:,0]+1j*acfd[rbin,:,1])
    noise_lev=1.6*max(1.,np.median(sorted(pwr0)[0:10]))
    print rbin, " Pwr0:",w[0]," Noise:",noise_lev
    more_badlags(w,good_lags_fluct[rbin],prm,noise_lev=noise_lev)
    #print "  More_bad:",rbin, np.sum(good_lags_fluct[rbin]),good_lags_fluct[rbin]

  tup=(good_lags_tx,good_lags_range,good_lags_fluct)
  good_lags=good_lags_tx & good_lags_range & good_lags_fluct
  return (good_lags,tup)


def fitacf_bad_lags(prm,pwr0,acfd):
  glags,tup=convo_good_lags(prm,pwr0,acfd)
  for item in tup:
    item=~item
  return ~glags,tup

#@profile
def convo_get_bad_lags(prm, txlags = None):
    if txlags == None:
        txlags = convo_good_lags_txsamples(prm)
    overlap_lags = convo_good_lags_overlap(prm) 
    return ~txlags | ~overlap_lags
