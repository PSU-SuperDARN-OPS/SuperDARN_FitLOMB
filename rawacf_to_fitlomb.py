# jon klein, jtklein@alaska.edu
# mit license
import argparse
import pdb
import h5py

from sd_data_tools import *
from pydmap import DMapFile, timespan, dt2ts, ts2dt
from complex_lomb import *
from timecube import TimeCube

import numpy as np
import matplotlib.pyplot as plt

from libfitacf import get_badsamples, get_badlags
from scipy.optimize import curve_fit
import scipy.signal as signal

from iterative_bayes import iterative_bayes
I_OFFSET = 0
Q_OFFSET = 1

DECAY_IDX = 0
POW_IDX = 1
DF_IDX = 2

MAX_V = 1000 # m/s, max velocity to search for 
C = 3e8

ce_matrix = []
se_matrix = []
CS_f = []

class LombFit:
    def __init__(self, record):
        self.rawacf = record # dictionary copy of RawACF record
        self.lags = range(self.rawacf['mplgs']) # range of lags
        self.nlags = self.rawacf['mplgs'] # number of lags
        self.ranges = range(self.rawacf['nrang']) # range gates
        self.nranges = self.rawacf['nrang'] # range gates
        self.pulses = self.rawacf['ptab'] # (mppul length list): pulse table
        self.npulses = self.rawacf['mppul'] # number of pulses
        self.ltab = self.rawacf['ltab'] # (mplgs x 2 length list): lag table
        self.t_first = self.rawacf['lagfr'] # lag to first range in us
        self.t_pulse = self.rawacf['mpinc'] # multi pulse increment (tau, basic lag time) 
        self.smsep = self.rawacf['smsep'] # sample separation in us
        self.txpl = self.rawacf['txpl'] # sample separation in us
        self.rsep = self.rawacf['rsep'] # range gate separation in km
        self.acfi = self.rawacf['acfd'][I_OFFSET::2]
        self.acfq = self.rawacf['acfd'][Q_OFFSET::2]
        self.slist = self.rawacf['slist'] 
        self.tfreq = self.rawacf['tfreq'] # transmit frequency (kHz)

        # TODO: copy over pwr0, ltab, ptab, slist, nlag
        # TODO:
        #       get widththreshold and peakthreshold.. 
            
        # calculate max frequency either nyquist rate, or calculated off max velocity
        fmax = (MAX_V * 2 * (self.tfreq * 1e3)) / C
        nyquist = 1 / (2e-6 * self.t_pulse)

        self.freqs = np.linspace(max(-nyquist, -fmax),min(nyquist, fmax), self.nlags * 30)
        self.maxwidth = 20
        self.widththreshold = .95 # widththreshold - statistical significance threshold for finding the extend of a peak
        self.peakthreshold = .5 # peakthreshold - statistical significance threshold for finding returns
        self.maxalf = 300
        self.alfsteps = 400
        
        # take average of smallest ten powers at range gate 0 for noise estimate
        self.noise = np.mean(sorted(self.rawacf['pwr0'])[:10])
        self.qflg = np.zeros(self.nranges)
        self.gflg = np.zeros(self.nranges)
        
        # initialize empty arrays for fitted parameters 
        self.lfits      = [[] for r in range(self.nranges)]
        self.sfits      = [[] for r in range(self.nranges)]

        self.v          = [[] for r in range(self.nranges)]
        self.v_e        = [[] for r in range(self.nranges)]
 
        self.sd_s       = [[] for r in range(self.nranges)]
        self.w_s_e      = [[] for r in range(self.nranges)]
        self.w_s        = [[] for r in range(self.nranges)]
        self.p_s        = [[] for r in range(self.nranges)]
        self.p_s_e      = [[] for r in range(self.nranges)]
        self.v_s        = [[] for r in range(self.nranges)]
        self.v_s_e      = [[] for r in range(self.nranges)]

        self.sd_l       = [[] for r in range(self.nranges)]
        self.w_l_e      = [[] for r in range(self.nranges)]
        self.w_l        = [[] for r in range(self.nranges)]
        self.p_l        = [[] for r in range(self.nranges)]
        self.p_l_e      = [[] for r in range(self.nranges)]
        self.v_l        = [[] for r in range(self.nranges)]
        self.v_l_e      = [[] for r in range(self.nranges)]

        self.CalcBadlags()

    # calculates the average noise the given lag (how?) 
    # find non-bad lags for each range
    # with real and complex values below limit (what limit?)
    # concatenate samples, calculate ACF
    def CalcSpectrumNoiseLevel(self):
        noise_samples = []
        for r in self.ranges:
            pass       


    # writes out a record of the lss fit
    def WriteLSSFit(self):
        pass         

    # calculates FitACF-like parameters for each peak in the spectrum
    # processes the pulse (move it __init__)?
    def ProcessPulse(self, cubecache):
        for r in self.ranges:
            if (r != 15):
                continue
            peaks = self.ProcessPeaks(r, cubecache)

    # finds returns in spectrum and records cell velocity
    def ProcessPeaks(self, rgate, cubecache):
        offset = self.nlags * rgate
        # acfd is mplgs * nrang
        # see http://davit.ece.vt.edu/davitpy/_modules/pydarn/sdio/radDataTypes.html
        
        i_lags = np.array(self.acfi[offset:offset+self.nlags])
        q_lags = np.array(self.acfq[offset:offset+self.nlags])
        # TODO: FIND GOOD LAGS MASK
        good_lags = np.ones(self.nlags)
        good_lags[self.bad_lags[rgate]] = False 

        lags = map(lambda x : abs(x[1]-x[0]),self.rawacf['ltab'])[0:self.nlags]


        #plt.subplot(2,1,1)
        #plt.plot(good_lags)
        #plt.subplot(2,1,2)
        #plt.plot(i_lags)
        #plt.plot(q_lags)


        i_lags = i_lags[good_lags == True]
        q_lags = q_lags[good_lags == True]

        t = (np.array(map(lambda x : abs(x[1]-x[0]),self.rawacf['ltab'])[0:self.nlags]) * self.t_pulse / 1e6)[good_lags == True]

        #plt.show()

        #np.arange(0, self.nlags * self.t_pulse, self.t_pulse)[good_lags == True] / 1e6
        
        samples = i_lags + 1j * q_lags
        alfs = np.linspace(0, self.maxalf, self.alfsteps)

        # calcuate generalized lomb-scargle periodogram iteratively
        self.lfits[rgate] = iterative_bayes(samples, t, self.freqs, alfs, cubecache, maxfreqs = 3, env_model = 1)
        #self.sfits[rgate] = iterative_bayes(samples, t, freqs, alfs, maxfreqs = 2, env_model = 2)
        

        # TODO: calculate qflg, gflg (ground and quality flags)
        self.qflg[rgate] = 0
        self.gflg[rgate] = 0

        # TODO: calculate v, v_e
    def ProcessPeak(self, fitsigma, fitlambda):
        # compute velocity, width, and power for each peak with "sigma" and "lambda" fits
        # TODO: add exception checking to handle fit failures
        # TODO: handle inf pcov (bad fits)
        # TODO: detect horrible fits (nonphysical spectral widths)
        # calculate "lambda" parameters

        self.sd_l[rgate].append(pcov[DECAY_IDX][DECAY_IDX]) # TODO: what is sd_l (standard deviation of lambda?)

        self.w_l[rgate].append(popt[DECAY_IDX])
        self.w_l_e[rgate].append(pcov[DECAY_IDX][DECAY_IDX])

        self.p_l[rgate].append(popt[POW_IDX])
        self.p_l_e[rgate].append(pcov[POW_IDX][POW_IDX])

        self.v_l[rgate].append(popt[DF_IDX] + freqs[peakidx])
        self.v_l_e[rgate].append(pcov[DF_IDX][DF_IDX])

        # calculate "sigma" parameters
        popt, pcov = curve_fit(gaussian, seg_freqs, seg_powers, p0 = [1,5,0], maxfev=2000)

        self.sd_l[rgate].append(pcov[DECAY_IDX][DECAY_IDX]) # TODO: what is sd_l (standard deviation of sigma?)

        self.w_s[rgate].append(popt[DECAY_IDX])
        self.w_s_e[rgate].append(pcov[DECAY_IDX][DECAY_IDX])

        self.p_s[rgate].append(popt[POW_IDX])
        self.p_s_e[rgate].append(pcov[POW_IDX][POW_IDX])

        self.v_s[rgate].append(popt[DF_IDX] + freqs[peakidx])
        self.v_s_e[rgate].append(pcov[DF_IDX][DF_IDX])

    def PlotLomb(self):
        # see http://matplotlib.org/examples/pylab_examples/pcolor_demo.html
        x = self.freqs # dwvelocity/frequency 
        y = self.ranges # ranges
        z = 0 # certainty 
        plt.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
        plt.axis([x.min(), x.max(), y.min(), y.max()])
        plt.colorbar()
        plt.show()

    def CalcBadlags(self):
        #bad_samples = get_badsamples(dfilet)
        bad_lags = [[] for i in range(self.nranges)]

        for i in range(self.nranges):
            bad_lags[i] = get_badlags(self.rawacf, i)
        
        self.bad_lags = bad_lags


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processes RawACF files with a Lomb-Scargle periodogram to produce FitACF-like science data.')
    
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--infile", help="input RawACF file to convert")
    parser.add_argument("--outfile", help="output FitLSS file")
    
    args = parser.parse_args() 
    # on 3/20/2013, 8 AM UTC
    infile = '20130320.0801.00.mcm.a.rawacf'
    dfile = DMapFile(files=[infile])

    times = dfile.times
    i = 0
    cubecache = TimeCube()

    for t in times:
        if(dfile[t]['bmnum'] != 9):
            continue
        fit = LombFit(dfile[t])
        fit.ProcessPulse(cubecache)
        i += 1
        if i > 40:
            break
    del dfile


