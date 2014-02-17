# jon klein, jtklein@alaska.edu
# mit license
import argparse
import pdb
import h5py

from sd_data_tools import *
from pydmap import DMapFile, timespan, dt2ts, ts2dt
from complex_lomb import *
import numpy as np
import matplotlib.pyplot as plt

from libfitacf import get_badsamples, get_badlags
from scipy.optimize import curve_fit
import scipy.signal as signal

I_OFFSET = 0
Q_OFFSET = 1

DECAY_IDX = 0
POW_IDX = 1
DF_IDX = 2

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
        self.acfd = self.rawacf['acfd'] # acf values
        
        # TODO: copy over pwr0, ltab, ptab, slist, nlag
        # TODO:
        #       get widththreshold and peakthreshold.. 

        self.freqs = np.linspace(-1./(2 * self.t_pulse), 1./(2 * self.t_pulse), self.nlags * 8) * 1e6
        self.maxwidth = 20
        self.widththreshold = .95 # widththreshold - statistical significance threshold for finding the extend of a peak
        self.peakthreshold = .5 # peakthreshold - statistical significance threshold for finding returns

        self.qflg = np.zeros(self.nranges)
        self.gflg = np.zeros(self.nranges)

        # initialize empty arrays for fitted parameters 
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

        self.ProcessPulse()
    
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
    def ProcessPulse(self):
        for r in self.ranges:
            peaks = self.ProcessPeaks(r)

    # finds returns in spectrum and records cell velocity
    def ProcessPeaks(self, rgate):
        offset = rgate * self.nlags * 2
        
        # acfd is mplgs * nrang
        # see http://davit.ece.vt.edu/davitpy/_modules/pydarn/sdio/radDataTypes.html
        i_lags = np.array(self.acfd[offset + I_OFFSET : offset+self.nlags * 2 : 2])
        q_lags = np.array(self.acfd[offset + Q_OFFSET : offset+self.nlags * 2 : 2])
        
        # TODO: FIND GOOD LAGS MASK
        good_lags = np.ones(self.nlags)
        
        i_lags = i_lags[good_lags == True]
        q_lags = q_lags[good_lags == True]
        t = np.arange(0, self.nlags * self.t_pulse, self.t_pulse)[good_lags == True] / 1e6
        
        samples = i_lags + 1j * q_lags

        # calcuate lomb-scargle periodogram
        lomb_range = lomb(samples, t, self.freqs)
        prob_range = lomb_ztop(lomb_range, self.freqs)
        lomb_range *= ((2 * np.std(samples)) / len(self.freqs)) # de-normalize spectrum

        # find peaks (assuming "well separated" peaks...)
        threshidxs = np.arange(len(self.freqs))[prob_range < self.peakthreshold]
        peakidxs = signal.find_peaks_cwt(lomb_range, np.arange(1,self.maxwidth))
        threshpeaksidxs = np.intersect1d(threshidxs, peakidxs)
       
        # calculate freq and prob 
        threshpeakfreqs = self.freqs[threshpeaksidxs]
        threshpeakprobs = prob_range[threshpeaksidxs]
        threshwidth = prob_range < self.widththreshold

        # calculate v, v_e
        self.v[rgate] = threshpeakfreqs # TODO: apply doppler shift to calculate velocity (need TF?)
        self.v_e[rgate] = threshpeakprobs # TODO: scale significance to some other error

        # TODO: calculate qflg, gflg (ground and quality flags)
        self.qflg[rgate] = 0
        self.gflg[rgate] = 0

        # unsupported: sd_phi
        plt.subplot(3,1,1)
        plt.plot(i_lags)
        plt.plot(q_lags)
        plt.subplot(3,1,2)
        plt.plot(self.freqs, lomb_range)
        plt.subplot(3,1,3)
        plt.plot(self.freqs, prob_range)
        plt.show()
        #for peakidx in peakidxs:
        #    # extract peak
        #    pdb.set_trace()
        #    seg_idxs = get_segment(threshwidth, peakidx)
        #    seg_freqs = self.freqs[seg_idxs] - self.freqs[peakidx]
        #    seg_powers = lomb_range[seg_idxs]
        #    self.ProcessPeaks(rgate, peakidx, seg_idxs, seg_powers)

    # processes a peak for sigma and lambda fit
    def ProcessPeak(self, rgate, peakidx, seg_idxs, seg_powers):
        # compute velocity, width, and power for each peak with "sigma" and "lambda" fits
        # TODO: feed better initial fit parameters to curve_fit
        # TODO: add exception checking to handle fit failures
        # TODO: handle inf pcov (bad fits)
        # TODO: detect horrible fits (nonphysical spectral widths)
        # calculate "lambda" parameters
        popt, pcov = curve_fit(lorentzian, seg_freqs, seg_powers, p0 = [1,5,0], maxfev=2000)

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

    def CalcBadlags(drange):
        # https://github.com/jspaleta/SuperDARN_MSI_ROS/blob/d925b7740e6a95c0e1b653a08bf8967361eb46f8/linux/home/radar/ros.3.6/codebase/superdarn/src.lib/tk/fitacf.2.5/src.lib/tk/fitacf.2.5/src/badlags.c
        # see FitACFBadlags and FitACFCkRng
        # https://web.archive.org/web/20110411212317/http://mediawiki.hmo.ac.za/mediawiki/index.php/FitACF
        
        # find crossrange interference

        # find samples which overlap with transmitted pulses 
        badsamples = [] 
        badlags = []
        rangesamp = (self.rsep * 20) / (3 * self.smsep) # calculate samples for each range gate
        n_samples = max(self.pulses) * t_pulse / self.smsep + self.nranges * rangesamp  # may not cover all samples.. but enough for the pulse sequence
        sample_times = [self.smsep * s for s in range(n_samples)]

        for p in pulses: 
            t1 = p * self.t_pulse - self.txpl/2.
            t2 = t1 + 3 * self.txpl / 2. + 100

            for ts in sample_times:
                ts = r * smsep
                
                if ts >= t1 and ts <= t2:
                    badsamples.append((ts - self.t_first)/self.smsep) # add bad sample index 

        badsamples = sorted(list(set(badsamples))) # remove duplicates, and sort list of bad samples
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processes RawACF files with a Lomb-Scargle periodogram to produce FitACF-like science data.')
    
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--infile", help="input RawACF file to convert")
    parser.add_argument("--outfile", help="output FitLSS file")
    
    args = parser.parse_args() 

    infile = '20140204.2200.04.kod.c.rawacf'
    dfile = DMapFile(files=[infile])

    times = dfile.times
    i = 0
    for t in times:
        i += 1
        if i > 20:
            pdb.set_trace()
        fit = LombFit(dfile[t])


    del dfile

    pdb.set_trace()
