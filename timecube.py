# jon klein, jtklein@alaska.edu
# 02/2014

# functions to cache calculations for lomb fits
# mostly time cubes.. cubes of amplitude(time,frequency,decay)
# recomputing these takes up ~80% of runtime if not cached

import numpy as np
import datetime
import pdb

# TimeCube is a class of cached cubes of amplitude(time, frequency, decay)
# creating these is expensive and RAM is cheap, so...

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

# https://stackoverflow.com/questions/13031439/element-wise-function-on-pycudacomplex-array
# need to use complex 1d array


