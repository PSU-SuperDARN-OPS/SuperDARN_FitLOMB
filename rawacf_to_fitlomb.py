# jon klein, jtklein@alaska.

# mit license
import argparse
import pdb

from sd_data_tools import *
from pydmap import DMapFile, timespan, dt2ts, ts2dt
from complex_lomb import *

from timecube import TimeCube, make_spacecube

import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates

from libfitacf import get_badsamples, get_badlags
from scipy.optimize import curve_fit
import scipy.signal as signal
import pp

from iterative_bayes import iterative_bayes, find_fwhm, calculate_bayes

I_OFFSET = 0
Q_OFFSET = 1

DECAY_IDX = 0
POW_IDX = 1
DF_IDX = 2

MAX_V = 1500 # m/s, max velocity to search for 
C = 3e8

VEL_CMAP = plt.cm.PuOr
FREQ_CMAP = plt.cm.spectral
NOISE_CMAP = plt.cm.autumn
SPECW_CMAP = plt.cm.hsv
POWER_CMAP = plt.cm.jet

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
        self.bmnum = self.rawacf['bmnum'] # beam number
        self.recordtime = datetime.datetime(self.rawacf['time.yr'], self.rawacf['time.mo'], self.rawacf['time.dy'], self.rawacf['time.hr'], self.rawacf['time.mt'], self.rawacf['time.sc'], self.rawacf['time.us']) 
        
        # TODO: copy over pwr0, ltab, ptab, slist, nlag
        # TODO:
        #       get widththreshold and peakthreshold.. 
            
        # calculate max frequency either nyquist rate, or calculated off max velocity
        fmax = (MAX_V * 2 * (self.tfreq * 1e3)) / C
        nyquist = 1 / (2e-6 * self.t_pulse)

        self.freqs = np.linspace(max(-nyquist, -fmax),min(nyquist, fmax), self.nlags * 10)
        self.maxwidth = 20
        self.widththreshold = .95 # widththreshold - statistical significance threshold for finding the extend of a peak
        self.peakthreshold = .5 # peakthreshold - statistical significance threshold for finding returns
        self.maxalf = 230
        self.alfsteps = 150
        self.maxfreqs = 3
        self.alfs = np.linspace(0, self.maxalf, self.alfsteps)

        # thresholds on velocity and spectral width for surface scatter flag (m/s)
        self.vss_thresh = 40
        self.wss_thresh = 40
        
        # threshold on power (snr), spectral width fwhm, and velocity fwhm for quality flag
        self.qwle_thresh = 15
        self.qvle_thresh = 15
        self.qpwr_thresh = 2
    
        # thresholds on velocity and spectral width for ionospheric scatter flag (m/s)
        self.wimin_thresh = 100
        self.wimax_thresh = 1400
        self.vimax_thresh = 1000
        self.vimin_thresh = 100

        # take average of smallest ten powers at range gate 0 for noise estimate
        self.noise = np.mean(sorted(self.rawacf['pwr0'])[:10])
        
        # initialize empty arrays for fitted parameters 
        self.lfits      = [[] for r in self.ranges]
        self.sfits      = [[] for r in self.ranges]

        self.v          = np.zeros([self.nranges, self.maxfreqs])
        self.v_e        = np.zeros([self.nranges, self.maxfreqs])
 
        self.sd_s       = np.zeros([self.nranges, self.maxfreqs])
        self.w_s_e      = np.zeros([self.nranges, self.maxfreqs])
        self.w_s        = np.zeros([self.nranges, self.maxfreqs])
        self.p_s        = np.zeros([self.nranges, self.maxfreqs])
        self.p_s_e      = np.zeros([self.nranges, self.maxfreqs])
        self.v_s        = np.zeros([self.nranges, self.maxfreqs])
        self.v_s_e      = np.zeros([self.nranges, self.maxfreqs])

        self.sd_l       = np.zeros([self.nranges, self.maxfreqs])
        self.w_l_e      = np.zeros([self.nranges, self.maxfreqs])
        self.w_l        = np.zeros([self.nranges, self.maxfreqs])
        self.p_l        = np.zeros([self.nranges, self.maxfreqs])
        self.p_l_e      = np.zeros([self.nranges, self.maxfreqs])
        self.v_l        = np.zeros([self.nranges, self.maxfreqs])
        self.v_l_e      = np.zeros([self.nranges, self.maxfreqs])

        self.gflg       = np.zeros([self.nranges, self.maxfreqs])
        self.iflg       = np.zeros([self.nranges, self.maxfreqs])
        self.qflg       = np.zeros([self.nranges, self.maxfreqs])
        
        self.v_l_e      = np.zeros([self.nranges, self.maxfreqs])

        self.CalcBadlags()

    # writes out a record of the lss fit
    def WriteLSSFit(self):
        pass         

    # calculates FitACF-like parameters for each peak in the spectrum
    # processes the pulse (move it __init__)?
    def ProcessPulse(self, cubecache):
        for r in self.ranges:
            peaks = self.CalculatePeaks(r, cubecache)
        self.ProcessPeaks()
    
    # TODO: work with non-lambda env models
    def ParallelProcessPulse(self):
        # create pp job server
        job_server = pp.Server()#ppservers=("137.229.27.61",""))

        # prepare sample and time arrays 
        times_samples = [(self._CalcSamples(r)) for r in self.ranges]
        
        jobs = []
        
        # dispatch jobs
        libs = ("numpy as np", "numexpr as ne", "timecube", "iterative_bayes")
        funcs = (make_spacecube, find_fwhm, calculate_bayes)

        for r in self.ranges:
            args = (times_samples[r][1], times_samples[r][0], self.freqs, self.alfs, 1, self.maxfreqs)
            jobs.append(job_server.submit(iterative_bayes, args, funcs, libs))
        
        # wait for jobs to complete
        job_server.wait() 
        
        for (rgate, job) in enumerate(jobs):
            try:
                self.lfits[rgate] = job()
            except:
                pdb.set_trace()
        job_server.print_stats()
        job_server.destroy()         
        self.ProcessPeaks()

    # get time and good complex samples for a range gate
    def _CalcSamples(self, rgate):
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
        samples = i_lags + 1j * q_lags
        return t, samples


    # finds returns in spectrum and records cell velocity
    def CalculatePeaks(self, rgate, cubecache, env_model = 1):
        t, samples = self._CalcSamples(rgate)

        # calcuate generalized lomb-scargle periodogram iteratively
        self.lfits[rgate] = iterative_bayes(samples, t, self.freqs, self.alfs, env_model, self.maxfreqs, cubecache = cubecache)
        #self.sfits[rgate] = iterative_bayes(samples, t, freqs, alfs, maxfreqs = 2, env_model = 2)
        
    def ProcessPeaks(self):
        # compute velocity, width, and power for each peak with "sigma" and "lambda" fits
        # TODO: add exception checking to handle fit failures
        for rgate in self.ranges:
            for (i, fit) in enumerate(self.lfits[rgate]):
                # calculate "lambda" parameters
                np.append(self.sd_l[rgate],0) # TODO: what is sd_l (standard deviation of lambda?)
                
                # see Effects of mixed scatter on SuperDARN convection maps near (1) for spectral width 
                self.w_l[rgate,i] = C / (2 * np.pi * fit['alpha'] * self.tfreq)
                self.w_l_e[rgate,i] = fit['alpha_fwhm']
                self.p_l[rgate,i] = fit['amplitude'] / self.noise
                self.p_l_e[rgate,i] = 0
                v_l = (fit['frequency'] * C) / (2 * self.tfreq * 1e3)
                self.v_l[rgate,i] = v_l
                self.v_l_e[rgate,i] = fit['frequency_fwhm']
                
                # set gflg if any of the returns match vss_thresh and wss_thresh thresholds
                # ss - surface scatter, v - velocity, w - spectral width
                if v_l < self.vss_thresh and self.w_l[rgate, i] < self.wss_thresh:
                    self.gflg[rgate,i] = 1
                
                # set iflg if ionospheric scatter velocity and spectral width thresholds are met
                if v_l < self.vimax_thresh and v_l > self.vimin_thresh and self.w_l[rgate,i] > self.wimin_thresh and self.w_l[rgate, i] < self.wimax_thresh:
                    self.iflg[rgate,i] = 1

                # set qflg if .. signal to noise ratios are high enough, not stuck 
                if self.p_l[rgate,i] > self.qpwr_thresh and self.w_l_e[rgate,i] < self.qwle_thresh and self.w_l_e[rgate, i] < self.qvle_thresh:
                    self.qflg[rgate,i] = 1

            '''
            for (i, fit) in enum(self.sfits[rgate]):
                # calculate "sigma" parameters
                self.sd_l[rgate].append(pcov[DECAY_IDX][DECAY_IDX]) # TODO: what is sd_l (standard deviation of sigma?)

                self.w_s[rgate].append(popt[DECAY_IDX])
                self.w_s_e[rgate].append(pcov[DECAY_IDX][DECAY_IDX])

                self.p_s[rgate].append(popt[POW_IDX])
                self.p_s_e[rgate].append(pcov[POW_IDX][POW_IDX])
            '''

# TODO: determine meaning for v_s, and v_s_e.. pick highest snr velocity for gate?
#        self.v_s[rgate].append(popt[DF_IDX] + freqs[peakidx])
#        self.v_s_e[rgate].append(pcov[DF_IDX][DF_IDX])
           
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
# plot w_l, p_l, and v_ms of main peak as a function of time
# adapted from jef's plot-rti.py
def PlotRTI(lombfits, beam):
    # assemble pulse time list
    times = [fit.recordtime for fit in lombfits]
    maxfreqs = lombfits[0].maxfreqs

    velocity = np.zeros([len(times), lombfits[0].nranges])
    for i in range(maxfreqs):
        plt.subplot(maxfreqs, 1, i+1)
        # assemble velocity list, collect velocity at beam number
        for (t,pulse) in enumerate(lombfits):
            if pulse.bmnum != beam:
                continue
            velocity[t,:] = pulse.v_l[:,i] * pulse.qflg[:,i] 

            ionospheric_scatter = np.sum(pulse.iflg * pulse.qflg, axis=1) > 0
            surface_scatter = np.sum(pulse.gflg * pulse.qflg, axis=1) > 0
            mixed_scatter = ionospheric_scatter * surface_scatter
             
            if sum(mixed_scatter):
                print 'mixed scatter at: ' + str( mixed_scatter.nonzero())


            
        
        x = dates.date2num(times)
        y = np.array(lombfits[0].ranges)
        
        plt.pcolor(x, y, velocity.T, cmap = VEL_CMAP)

        plt.axis([x.min(), x.max(), y.min(), y.max()])
        #plt.axis([x.min(), x.max(), y.min(), y.max()])
        plt.grid(True)
        plt.colorbar()
        ax = plt.gca()

        locator = dates.AutoDateLocator()
        dateformatter = dates.AutoDateFormatter(locator)

        ax.xaxis.set_major_locator(locator) 
        ax.xaxis.set_major_formatter(dateformatter)
    plt.show()

    pdb.set_trace()
    #fig = plt.gcf() 
    #fig.autofmt_xdate()

        
    
    # prepare     

        # see http://matplotlib.org/examples/pylab_examples/pcolor_demo.html
        #x = self.freqs # dwvelocity/frequency 
        #y = self.ranges # ranges
        #z = 0 # certainty 
        #plt.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
        #plt.axis([x.min(), x.max(), y.min(), y.max()])
        #plt.colorbar()
        #plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processes RawACF files with a Lomb-Scargle periodogram to produce FitACF-like science data.')
    
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--infile", help="input RawACF file to convert")
    parser.add_argument("--outfile", help="output FitLSS file")
    
    args = parser.parse_args() 

    # good time at McM is 3/20/2013, 8 AM UTC
    infile = '/mnt/flash/sddata/0208/all.rawacf'
    #20130320.0801.00.mcm.a.rawacf'#'20130320.0801.00.mcm.a.rawacf'
    dfile = DMapFile(files=[infile])

    times = dfile.times
    cubecache = TimeCube()
    
    lombfits = []
    for (i,t) in enumerate(times):
        if(dfile[t]['bmnum'] != 9):
            continue
        print i
        print 'processing time ' + str(t)
        fit = LombFit(dfile[t])

        fit.ParallelProcessPulse()
        lombfits.append(fit)
    PlotRTI(lombfits, 9)
    
