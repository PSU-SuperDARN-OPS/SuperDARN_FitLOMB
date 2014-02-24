# jon klein, jtklein@alaska.edu
# mit license
import argparse
import pdb

from sd_data_tools import *
from pydmap import DMapFile, timespan, dt2ts, ts2dt
from complex_lomb import *

from timecube import TimeCube

import datetime
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

MAX_V = 1500 # m/s, max velocity to search for 
C = 3e8

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
        
        self.recordtime = datetime.datetime(self.rawacf['time.yr'], self.rawacf['time.mo'], self.rawacf['time.dy'], self.rawacf['time.hr'], self.rawacf['time.mt'], self.rawacf['time.sc'], self.rawacf['time.us']) 
        pdb.set_trace()
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

    # writes out a record of the lss fit
    def WriteLSSFit(self):
        pass         

    # calculates FitACF-like parameters for each peak in the spectrum
    # processes the pulse (move it __init__)?
    def ProcessPulse(self, cubecache):
        for r in self.ranges:
            peaks = self.CalculatePeaks(r, cubecache)
        #self.ProcessPeaks()

    # finds returns in spectrum and records cell velocity
    def CalculatePeaks(self, rgate, cubecache):
        offset = self.nlags * rgate
        # acfd is mplgs * nrang
        # see http://davit.ece.vt.edu/davitpy/_modules/pydarn/sdio/radDataTypes.html
        
        i_lags = np.array(self.acfi[offset:offset+self.nlags])
        q_lags = np.array(self.acfq[offset:offset+self.nlags])
        # TODO: FIND GOOD LAGS MASK
        good_lags = np.ones(self.nlags)
        good_lags[self.bad_lags[rgate]] = False 

        lags = map(lambda x : abs(x[1]-x[0]),self.rawacf['ltab'])[0:self.nlags]

        i_lags = i_lags[good_lags == True]
        q_lags = q_lags[good_lags == True]

        t = (np.array(map(lambda x : abs(x[1]-x[0]),self.rawacf['ltab'])[0:self.nlags]) * self.t_pulse / 1e6)[good_lags == True]

        #np.arange(0, self.nlags * self.t_pulse, self.t_pulse)[good_lags == True] / 1e6
        print 'noise: ' + str(self.noise)
        samples = i_lags + 1j * q_lags
        alfs = np.linspace(0, self.maxalf, self.alfsteps)

        # calcuate generalized lomb-scargle periodogram iteratively
        self.lfits[rgate] = iterative_bayes(samples, t, self.freqs, alfs, cubecache, maxfreqs = 3, env_model = 1)
        #self.sfits[rgate] = iterative_bayes(samples, t, freqs, alfs, maxfreqs = 2, env_model = 2)
        
    def ProcessPeaks(self):
        # compute velocity, width, and power for each peak with "sigma" and "lambda" fits
        # TODO: add exception checking to handle fit failures
        # TODO: detect horrible fits (nonphysical spectral widths)
        for rgate in self.ranges:
            for (i, fit) in enum(self.lfits[rgate]):
                # calculate "lambda" parameters
                self.sd_l[rgate].append(pcov[DECAY_IDX][DECAY_IDX]) # TODO: what is sd_l (standard deviation of lambda?)

                self.w_l[rgate].append(1/fit['alpha'])
                self.w_l_e[rgate].append(1/fit['alpha_fwhm'])

                self.p_l[rgate].append(fit['amplitude'] / self.noise)
                self.p_l_e[rgate].append(0)

                self.v_l[rgate].append(0)
                self.v_l_e[rgate].append(pcov[DF_IDX][DF_IDX])

            for (i, fit) in enum(self.sfits[rgate]):
                # calculate "sigma" parameters
                self.sd_l[rgate].append(pcov[DECAY_IDX][DECAY_IDX]) # TODO: what is sd_l (standard deviation of sigma?)

                self.w_s[rgate].append(popt[DECAY_IDX])
                self.w_s_e[rgate].append(pcov[DECAY_IDX][DECAY_IDX])

                self.p_s[rgate].append(popt[POW_IDX])
                self.p_s_e[rgate].append(pcov[POW_IDX][POW_IDX])

# TODO: determine meaning for v_s, and v_s_e.. pick highest snr velocity for gate?
#        self.v_s[rgate].append(popt[DF_IDX] + freqs[peakidx])
#        self.v_s_e[rgate].append(pcov[DF_IDX][DF_IDX])

        # TODO: calculate qflg, gflg (ground and quality flags)
        # TODO: calculate v, v_e
        self.qflg[rgate] = 0 
        self.gflg[rgate] = 0


    def PlotPeak(self, rgate):
        for (i,fit) in enumerate(self.lfits[rgate]):
            plt.subplot(len(self.lfits[rgate]),1,i+1)

            samples = fit['samples'] / self.noise
            fitsignal = fit['signal'] / self.noise
            t = fit['t']

            plt.plot(t,np.real(samples),'-.',color='r',linewidth=3)
            plt.plot(t,np.imag(samples),'-.',color='b',linewidth=3)
            plt.plot(t,np.real(fitsignal),color='r',linewidth=3)
            plt.plot(t,np.imag(fitsignal),color='b',linewidth=3)
            plt.grid(True)

            vel = (fit['frequency'] * 3e8) / (2 * self.tfreq * 1e3)

            plt.title('pass ' + str(i) + ' velocity (m/s): ' + str(vel) + 'freq fwhm: ' + str(fit['frequency_fwhm']))
            plt.legend(['I samples', 'Q samples', 'I fit', 'Q fit'])
            plt.xlabel('time (seconds)')
            plt.ylabel('SNR')

        plt.show()

    def CalcBadlags(self):
        bad_lags = [[] for i in range(self.nranges)]

        for i in range(self.nranges):
            bad_lags[i] = get_badlags(self.rawacf, i)
        
        self.bad_lags = bad_lags

# replicate rti-style plot 
# plot p_l and v_ms of main peak as a function of time
# adapted from jef's plot-rti.py
def PlotRTI(LombFits, beam):
    # assemble pulse time list
    times = [fit.recordtime for fit in LombFits]
    
    # assemble velocity list
    for 
    
    # prepare     

        # see http://matplotlib.org/examples/pylab_examples/pcolor_demo.html
        #x = self.freqs # dwvelocity/frequency 
        #y = self.ranges # ranges
        #z = 0 # certainty 
        #plt.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
        #plt.axis([x.min(), x.max(), y.min(), y.max()])
        #plt.colorbar()
        #plt.show()
'''
    plt.title("%(radar)s channel %(channel)s :: RTI Plot of %(plotvar1)s\n" % (plotdict) + "on %d/%d/%d\nAlong Beam Direction: %s\n" %
                                 (startday.month,startday.day,startday.year,text,))
        clocator=MaxNLocator(nbins=4)
        c1b = matplotlib.colorbar.ColorbarBase(
               c1ax, cmap=plot1_cmap,norm=norm1,orientation='vertical',ticks=clocator)
        c2b = matplotlib.colorbar.ColorbarBase(
               c2ax, cmap=plot2_cmap,norm=norm2,orientation='vertical',ticks=clocator)
        c1b.set_label(plotdict["plotvar1"]+" "+plotdict["plotvar1_label"])
        c2b.set_label(plotdict["plotvar2"]+" "+plotdict["plotvar2_label"])
        tcax.xaxis.set_ticks_position('bottom')
        tb = matplotlib.colorbar.ColorbarBase(
               tcax, cmap=tfreq_cmap,norm=tnorm,
               orientation='horizontal',ticks=[9,14,19])
        tcax.xaxis.set_label_text("Freq [MHz]",fontsize=6,va='center',ha='right')
        tcax.xaxis.set_label_coords(-0.2,0.5)
        labels=tcax.xaxis.get_majorticklabels()
        for label in labels: label.set_fontsize(6)
          #tcax.tick_params(axis='x', labelsize=6)
    #              bcax.xaxis.set_ticks_position('top')
    #              bb = matplotlib.colorbar.ColorbarBase(
    #                   bcax, cmap=beam_cmap,norm=bnorm,
    #                   orientation='horizontal',ticks=[0,1,2,3,4])
    #              bcax.xaxis.set_label_text("Beam",fontsize=6,va='center',ha='left')
    #              bcax.xaxis.set_label_coords(1.2,0.5)
    #              #bcax.tick_params(axis='x', labelsize=6)
    #              labels=bcax.xaxis.get_majorticklabels()
    #              for label in labels: label.set_fontsize(6)  
    #              bcax.xaxis.get_offset_text().set_visible(False) 

        ncax.xaxis.set_ticks_position('top')
        nformat=p.ScalarFormatter(useOffset=False)
        nformat.set_powerlimits((0,0))
        nb = matplotlib.colorbar.ColorbarBase(
               ncax, cmap=noise_cmap,norm=nnorm,format=nformat,
               orientation='horizontal',ticks=[0,5000,10000])
        ncax.xaxis.set_label_text("Noise [x1E4]",fontsize=6,va='center',ha='left')
        ncax.xaxis.set_label_coords(1.2,0.5)
        #ncax.tick_params(axis='x', labelsize=6,labeltop=True,labelbottom=False)
        labels=ncax.xaxis.get_majorticklabels()
        for label in labels: label.set_fontsize(6)
        ncax.xaxis.get_offset_text().set_visible(False)
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processes RawACF files with a Lomb-Scargle periodogram to produce FitACF-like science data.')
    
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--infile", help="input RawACF file to convert")
    parser.add_argument("--outfile", help="output FitLSS file")
    
    args = parser.parse_args() 

    # good time at McM is 3/20/2013, 8 AM UTC
    infile = '20130320.0801.00.mcm.a.rawacf'#'20130320.0801.00.mcm.a.rawacf'
    dfile = DMapFile(files=[infile])

    times = dfile.times
    cubecache = TimeCube()
    
    for (i,t) in enumerate(times):
        fit = LombFit(dfile[t])
        fit.ProcessPulse(cubecache)
        if i > 20:
            break
    
    PlotRTI(LombFits, 9):
    
