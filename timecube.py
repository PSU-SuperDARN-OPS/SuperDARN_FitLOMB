# jon klein, jtklein@alaska.edu
# 02/2014

# functions to cache calculations for lomb fits
# mostly time cubes.. cubes of amplitude(time,frequency,decay)
# recomputing these takes up ~80% of runtime if not cached

import numpy as np
import datetime

# TimeCube is a class of cached cubes of amplitude(time, frequency, decay)
# creating these is expensive and RAM is cheap, so...
# TODO: create generalized timecube with given function and inputs
# TODO: create fine resolution timecube, then provide masked timecubes..
# input: t, f, alfs, env_model


class TimeCube:
    def __init__(self, maxsize = 20):
        self.cubecache = {}
        self.cubetimes = {}
        self.maxsize = maxsize
       
    def _cubeparam_key(self, t, f, alfs, env_model):
        # there must be better ways of doing this...
        return str(t) + self._listhash(f) + self._listhash(alfs) + str(env_model)

    def _listhash(self,l):
        return ','.join([str(l[0]), str(l[-1]), str(len(l))])

    def get_spacecube(self, t, f, alfs, env_model):
        key = self._cubeparam_key(t,f,alfs,env_model) 

        if not key in self.cubecache:
            self.cubecache[key] = make_spacecube(t, f, alfs, env_model)
            
            # check to see if cube cache is too big. it is, delete the least recently used cube
            self.cubetimes[datetime.datetime.now()] = key 
            if len(self.cubecache) > self.maxsize:
                del self.cubecache[self.cubetimes[min(self.cubetimes.keys())]]
                del self.cubetimes[min(self.cubetimes.keys())]
        else:
            self.cubetimes

        return self.cubecache[key] 

def make_spacecube(t, f, alfs, env_model):
    # create ce_matrix and se_matrix..
    # sin and cos(w * t) * exp(-alf * t) cubes 
    omegas = 2 * np.pi * f
    c_matrix = np.cos(np.outer(omegas, t))
    s_matrix = np.sin(np.outer(omegas, t))
    #print "space cube c_matrix:",c_matrix.shape,len(c_matrix)
    # create cube of time by frequency with no decay
    c_cube = np.tile(c_matrix, (len(alfs),1,1))
    s_cube = np.tile(s_matrix, (len(alfs),1,1))

    # about 80% of execution time spent here..
    # ~ 400000
    # kernprof.py -l iterative_bayes.py
    # python -m line_profiler iterative_bayes.py.lprof

    # so... create cube of alpha by time at dc 
    envelope = np.exp(np.outer(-(alfs ** env_model), t))
    envelope_cube = np.tile(envelope, (len(c_matrix), 1,1))

    # rearrange cubes to match format of ce_matrix and se_matrix
    envelope_cube = np.swapaxes(envelope_cube, 1, 2)
    c_cube = np.swapaxes(c_cube, 0, 1)
    c_cube = np.swapaxes(c_cube, 1, 2)
    s_cube = np.swapaxes(s_cube, 0, 1)
    s_cube = np.swapaxes(s_cube, 1, 2)
    #print "space cube c_cube:",c_cube.shape
    #print "space cube envelope:",envelope.shape
    #print "space cube envelope_cube:",envelope_cube.shape
    
    ce_matrix = c_cube * envelope_cube
    se_matrix = s_cube * envelope_cube

    # C_f and S_f don't vary with the samples, only the envelopes
    # also.. C_f = S_f for simultaenous samples
    # so, lets only calculate this once per set of alpha/frequency/timespan

    # calculate C_f and S_f (14) and (15) in [4]
    # simultaneous sampling, so C_f and S_f reduce to the total power in envelope model
    z_matrix = np.ones([len(omegas), len(alfs)])
    for (k, alf) in enumerate(alfs):
        z_matrix[:,k] *= sum(envelope[k,:] ** 2)

    CS_f = z_matrix.T

    return ce_matrix, se_matrix, CS_f

def make_hyperspacecube(tfreqs_hz,times_secs,fcrit_hz,Vlos_mps,alfs, env_model):
    # tfreqs_hz and times_secs are equal length.
    # fcrit_hz vlos_mps and alfs are parameters to be found.
    C_mps=299792458.
    
    # generate theta_matrix
    fcritcube = np.tile(fcrit_hz, (len(Vlos_mps), len(times_secs), 1))
    vloscube = np.tile(Vlos_mps, (len(fcrit_hz), len(times_secs),  1))
    tseccube = np.tile(times_secs, (len(Vlos_mps), len(fcrit_hz), 1))
    tfreqcube = np.tile(tfreqs_hz, (len(Vlos_mps), len(fcrit_hz), 1))

    vloscube = np.swapaxes(vloscube, 1, 2)
    vloscube = np.swapaxes(vloscube, 0, 1)
    fcritcube = np.swapaxes(fcritcube, 1, 2)

    ns = np.sqrt(1. -((fcritcube **2)/(tfreqcube ** 2)))
    vm = ns * vloscube
    theta_matrix = (2. * np.pi) * (2 * tfreqcube * vm / C_mps) * tseccube
   
    # create ce_matrix and se_matrix..
    # sin and cos(w * t) * exp(-alf * t) cubes 
    c_matrix = np.cos(theta_matrix)
    s_matrix = np.sin(theta_matrix)
    # C_matrix should have dims of : Vlos,fcrit, times
    #print "hypertimecube: c_matrix:",c_matrix.shape,len(c_matrix)

    # create cube of time by frequency with no decay
    c_cube = np.tile(c_matrix, (len(alfs),1,1,1))
    s_cube = np.tile(s_matrix, (len(alfs),1,1,1))
    # kernprof.py -l iterative_bayes.py
    # python -m line_profiler iterative_bayes.py.lprof

    # so... create cube of alpha by time at dc 
    envelope = np.exp(np.outer(-(alfs ** env_model), times_secs**env_model))
    envelope_cube = np.tile(envelope, (len(Vlos_mps),len(fcrit_hz),1,1))

    # rearrange cubes to match format of ce_matrix and se_matrix
    envelope_cube = np.swapaxes(envelope_cube, 2, 3)

    c_cube = np.swapaxes(c_cube, 0, 1)
    c_cube = np.swapaxes(c_cube, 1, 2)
    c_cube = np.swapaxes(c_cube, 2, 3)
    s_cube = np.swapaxes(s_cube, 0, 1)
    s_cube = np.swapaxes(s_cube, 1, 2)
    s_cube = np.swapaxes(s_cube, 2, 3)
    
    ce_matrix = c_cube * envelope_cube
    se_matrix = s_cube * envelope_cube
    # ce_matrix should have dims: Vlos,fcrit,times,alpha

    # C_f and S_f don't vary with the samples, only the envelopes
    # also.. C_f = S_f for simultaenous samples
    # so, lets only calculate this once per set of alpha/frequency/timespan

    # calculate C_f and S_f (14) and (15) in [4]
    # simultaneous sampling, so C_f and S_f reduce to the total power in envelope model
    z_matrix=np.ones((len(Vlos_mps),len(fcrit_hz),len(alfs)))

    for (k, alf) in enumerate(alfs):
        z_matrix[:,:,k] *= sum(envelope[k,:] ** 2)
    CS_f = z_matrix.T

    return ce_matrix, se_matrix, CS_f

if __name__ == '__main__':
    env_model = 1
    fs = 100.
    ts = 1./fs

    freqs = np.linspace(-fs/2, fs/2, 16)
    alfs = np.linspace(0,fs/2, 16)

    lags = np.arange(0, 24) * ts
    
    lmask = [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0]

    ce_matrix, se_matrix, CS_f = make_spacecube(lags, freqs, alfs, env_model)
    import pdb
    pdb.set_trace()
    print CS_f[:,0]
    ''' [ 24.          19.25207407  15.73828838  13.10322194  11.09979817
       9.55503758   8.34691265   7.38863827   6.61795938   5.98980989
          5.47125489   5.03798619   4.67187934   4.35927957   4.08979102
             3.85541536]
             '''
