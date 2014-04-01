# jon klein, jtklein@alaska.
# functions to calculate a fitlomb (generalized lomb-scargle peridogram) from a rawacf
# parallized with python pp, possible to parallelize over the network
# mit license

import argparse

from sd_data_tools import *
from pydmap import DMapFile, timespan, dt2ts, ts2dt
from complex_lomb import *

from timecube import TimeCube, make_spacecube

import datetime, calendar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import h5py

from libfitacf import get_badsamples, get_badlags
from scipy.optimize import curve_fit
import scipy.signal as signal
import pp

from iterative_bayes import iterative_bayes, find_fwhm, calculate_bayes, calc_zoomvar

FITLOMB_REVISION_MAJOR = 0
FITLOMB_REVISION_MINOR = 1
ORIGIN_CODE = 'rawacf_to_fitlomb.py'
DATA_DIR = './data/'
FITLOMB_README = 'This group contains data from one SuperDARN pulse sequence with Lomb-Scargle Periodogram fitting.'

I_OFFSET = 0
Q_OFFSET = 1

FWHM_TO_SIGMA = 2.355 # conversion of fwhm to std deviation, assuming gaussian
MAX_V = 1000 # m/s, max velocity (doppler shift) to include in lomb
MAX_W = 1200 # m/s, max spectral width to include in lomb 
C = 3e8

ALPHA_RES = 30 # m/s
VEL_RES = 30 # m/s

VEL_CMAP = plt.cm.RdBu
FREQ_CMAP = plt.cm.spectral
NOISE_CMAP = plt.cm.autumn
SPECW_CMAP = plt.cm.hot
POWER_CMAP = plt.cm.jet

BEAM_ATTRS = ['radar.revision.major', 'radar.revision.minor',\
        'origin.command', 'cp', 'stid', \
        'rawacf.revision.major', 'rawacf.revision.minor',\
        'intt.sc', 'intt.us', 'txpl', 'mpinc', 'mppul', 'mplgs', 'nrang', 'frang', 'rsep', 'xcf',\
        'rawacf.revision.major', 'rawacf.revision.minor']

GROUP_ATTRS = [\
        'time.yr', 'time.mo', 'time.dy', 'time.hr', 'time.mt', 'time.sc', 'time.us', \
        'txpow', 'nave', 'atten', 'lagfr', 'smsep', 'ercod', 'stat.agc', 'stat.lopwr', \
        'noise.search', 'noise.mean', 'channel', 'bmnum', 'bmazm', 'scan', 'offset', 'rxrise',\
        'tfreq', 'mxpwr', 'lvmax', 'combf']

GROUP_ATTR_TYPES = [\
        np.int16, np.int8, np.int8, np.int8, np.int8, np.int8, np.int32, \
        np.int16, np.int16, np.int16, np.int16, np.int16, np.int16, np.int8, np.int8,\
        np.float32, np.float32, np.int8, np.int8, np.float32, np.int16, np.int16, np.int16,\
        np.int16, np.int32, np.int32, str]

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
        
        # calculate max decay rate for MAX_W spectral width
        amax = (MAX_W) / (C / (self.tfreq * 1e3 * 2 * np.pi))

        # calculate max frequency either nyquist rate, or calculated off max velocity
        nyquist = 1 / (2e-6 * self.t_pulse)
        fmax = min((MAX_V * 2 * (self.tfreq * 1e3)) / C, nyquist)
        df = (VEL_RES * 2.) * (self.tfreq * 1e3) / C
        self.freqs = np.linspace(-fmax,fmax, 2. * fmax / df)
        
        self.maxalf = amax
        self.alfsteps = int(amax / ALPHA_RES)
        self.maxfreqs = 2
        self.alfs = np.linspace(0, self.maxalf, self.alfsteps)

        # thresholds on velocity and spectral width for surface scatter flag (m/s)
        self.v_thresh = 30.
        self.w_thresh = 90. # blanchard, 2009
        
        # threshold on power (snr), spectral width std error m/s, and velocity std error m/s for quality flag
        self.qwle_thresh = 90
        self.qvle_thresh = 90
        self.qpwr_thresh = 2
        
        # threshold (snr) to keep data
        self.pwr_keepthresh = 2

        # thresholds on velocity and spectral width for ionospheric scatter flag (m/s)
        self.wimin_thresh = 100
        self.wimax_thresh = 1100
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

        self.keep       = np.zeros([self.nranges, self.maxfreqs])
       
        self.nlag       = np.zeros([self.nranges])
        
        self.CalcBadlags()

    # appends a record of the lss fit to an hdf5 file
    def WriteLSSFit(self, hdf5file):
        # create a group for /[beam number]/[record time]
        groupname = str(self.bmnum) + '/' + str(self.recordtime)
        grp = hdf5file.create_group(groupname)
               
        # add scalars as attributes to group
        for (i,attr) in enumerate(GROUP_ATTRS):
            grp.attrs[attr] = GROUP_ATTR_TYPES[i](self.rawacf[attr])
        
        beamgrp = hdf5file[str(self.bmnum)]
        beamgrp.attrs['readme'] = FITLOMB_README
        beamgrp.attrs['fitlomb.revision.major'] = FITLOMB_REVISION_MAJOR 
        beamgrp.attrs['fitlomb.revision.minor'] = FITLOMB_REVISION_MINOR
        beamgrp.attrs['fitlomb.bayes.iterations'] = self.maxfreqs 
        beamgrp.attrs['origin.code'] = ORIGIN_CODE # TODO: ADD ARGUEMENTS
        beamgrp.attrs['origin.time'] = str(datetime.datetime.now())
        beamgrp.attrs['rawacf.origin.code'] = self.rawacf['origin.code']
        beamgrp.attrs['rawacf.origin.time'] = self.rawacf['origin.time']

        for gattr in BEAM_ATTRS:
            beamgrp.attrs[attr] = self.rawacf[gattr]
        
        grp.attrs['epoch.time'] = calendar.timegm(self.recordtime.timetuple()) + int(self.rawacf['time.us'])/1e6
        # TODO: SET THE FOLLOWING
        #grp.attrs['noise.sky'] = 0 # sky noise? 
        grp.attrs['noise.lag0'] = np.float32(self.noise) # lag zero power from noise acf?
        #grp.attrs['noise.vel'] = 0 # velocity from fitting noise acf?
        #grp.attrs['thr'] = 0 # wtf is this?
        
        # copy over vectors from rawacf
        add_compact_dset(hdf5file, groupname, 'ptab', np.int16(self.rawacf['ptab']), h5py.h5t.STD_I16BE)
        add_compact_dset(hdf5file, groupname, 'ltab', np.int16(self.rawacf['ltab']), h5py.h5t.STD_I16BE)
        add_compact_dset(hdf5file, groupname, 'slist', np.int16(self.rawacf['slist']), h5py.h5t.STD_I16BE)
        add_compact_dset(hdf5file, groupname, 'pwr0', np.float32(self.rawacf['pwr0']), h5py.h5t.NATIVE_FLOAT)
        
        # add calculated parameters
        add_compact_dset(hdf5file, groupname, 'qflg', np.int8(self.qflg), h5py.h5t.STD_I8BE)
        add_compact_dset(hdf5file, groupname, 'gflg', np.int8(self.gflg), h5py.h5t.STD_I8BE)
        add_compact_dset(hdf5file, groupname, 'iflg', np.int8(self.iflg), h5py.h5t.STD_I8BE)
        add_compact_dset(hdf5file, groupname, 'nlag', np.int16(self.nlag), h5py.h5t.STD_I16BE)
        
        add_compact_dset(hdf5file, groupname, 'keep_key', np.int16(self.keep), h5py.h5t.STD_I8BE)

        add_compact_dset(hdf5file, groupname, 'p_l', np.float32(self.p_l), h5py.h5t.NATIVE_FLOAT)
        add_compact_dset(hdf5file, groupname, 'p_l_e', np.float32(self.p_l_e), h5py.h5t.NATIVE_FLOAT)
        add_compact_dset(hdf5file, groupname, 'w_l', np.float32(self.w_l), h5py.h5t.NATIVE_FLOAT)
        add_compact_dset(hdf5file, groupname, 'w_l_e', np.float32(self.w_l_e), h5py.h5t.NATIVE_FLOAT)
        add_compact_dset(hdf5file, groupname, 'v_l', np.float32(self.v_l), h5py.h5t.NATIVE_FLOAT)
        add_compact_dset(hdf5file, groupname, 'v_l_e', np.float32(self.v_l_e), h5py.h5t.NATIVE_FLOAT)


        # TODO: verify that the following vectors are not relevant for fitlomb: phi0, phi0_e, sd_l, sd_s, sd_phi
        # add fitlomb specific attributtes and datasets
    

    # calculates FitACF-like parameters for each peak in the spectrum
    # processes the pulse (move it __init__)?
    def ProcessPulse(self, cubecache):
        for r in self.ranges:
            peaks = self.CalculatePeaks(r, cubecache)
        self.ProcessPeaks()
    
    # TODO: move FitACF compare function from plot_pickle to here
    def FitACFCompare(self, fitacffile):
        pass

    # TODO: work with non-lambda env models
    def ParallelProcessPulse(self, cubecache = False):
        # create pp job server
        job_server = pp.Server()#ppservers=("137.229.27.61",""))
        
        # prepare sample and time arrays 

        times_samples = [(self._CalcSamples(r)) for r in self.ranges]
        self.nlag[:] = [len(samps[0]) for samps in times_samples]
        
        jobs = []
        
        # dispatch jobs
        libs = ("numpy as np", "numexpr as ne", "timecube", "iterative_bayes")
        funcs = (make_spacecube, TimeCube, find_fwhm, calculate_bayes, calc_zoomvar)

        for r in self.ranges:
            args = (times_samples[r][1], times_samples[r][0], self.freqs, self.alfs, 1, self.maxfreqs, cubecache)
            jobs.append(job_server.submit(iterative_bayes, args, funcs, libs))
        
        # wait for jobs to complete
        job_server.wait() 
        
        for (rgate, job) in enumerate(jobs):
            self.lfits[rgate] = job()

        job_server.destroy() 

        self.ProcessPeaks()

    # get time and good complex samples for a range gate
    def _CalcSamples(self, rgate):
        offset = self.nlags * rgate

        # see http://davit.ece.vt.edu/davitpy/_modules/pydarn/sdio/radDataTypes.html
        i_lags = np.array(self.acfi[offset:offset+self.nlags])
        q_lags = np.array(self.acfq[offset:offset+self.nlags])
        
        good_lags = np.ones(self.nlags)
        good_lags[self.bad_lags[rgate] != 0] = 0

        lags = map(lambda x : abs(x[1]-x[0]),self.rawacf['ltab'])[0:self.nlags]

        i_lags = i_lags[good_lags == True]
        q_lags = q_lags[good_lags == True]

        t = (np.array(lags) * self.t_pulse / 1e6)[good_lags == True]
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
                np.append(self.sd_l[rgate],0) # TODO: what is a reasonable value for this? 
                 
                # see Effects of mixed scatter on SuperDARN convection maps near (1) for spectral width 
                # see ros.3.6/codebase/superdarn/src.lib/tk/fitacf/src/fit_acf.c and do_fit.c
                #self.w_l[rgate,i] = (C * fit['alpha']) / (2. * np.pi * (self.t_pulse * 1e-6) * (self.tfreq * 1e3)) 
                self.w_l[rgate,i] = (C * fit['alpha']) / (2. * np.pi * (self.tfreq * 1e3)) 

                # approximate alpha error by taking half of range of alphas covered in fwhm
                self.w_l_e[rgate,i] = fit['alpha_fwhm'] / FWHM_TO_SIGMA
                
                # amplitude estimation, see bayesian analysis v: amplitude estimation, multiple well separated sinusoids
                # bretthorst, equation 78, I'm probably doing this wrong...
                # to match fitacf, scale p_l by 10 * log10
                self.p_l[rgate,i] = fit['amplitude'] / self.noise
                self.p_l_e[rgate,i] = np.sqrt((self.noise ** 2)/fit['amplitude_error_unscaled'])/self.noise # this may be scaled wrong..

                v_l = (fit['frequency'] * C) / (2 * self.tfreq * 1e3)
                self.v_l[rgate,i] = v_l
                # approximate velocity error as half the range of velocities covered by fwhm 
                # "nonuniform sampling: bandwidth and aliasing", page 25
                self.v_l_e[rgate,i] = (((fit['frequency_fwhm']) * C) / (2 * self.tfreq * 1e3)) / FWHM_TO_SIGMA
                
                # for information on setting surface/ionospheric scatter thresholds, see
                # A new approach for identifying ionospheric backscatterin midlatitude SuperDARN HF radar observations
                if abs(v_l) - (self.v_thresh - (self.v_thresh / self.w_thresh) * abs(self.w_l[rgate, i])) > 0: 
                    self.iflg[rgate,i] = 1
                else:
                    self.gflg[rgate,i] = 1
                
                # set qflg if .. signal to noise ratios are high enough, not stuck 
                if self.p_l[rgate,i] > self.qpwr_thresh and \
                        self.w_l_e[rgate,i] < self.qwle_thresh and \
                        self.v_l_e[rgate, i] < self.qvle_thresh and \
                        self.w_l[rgate, i] < self.wimax_thresh and \
                        self.v_l[rgate, i] < self.vimax_thresh and \
                        self.w_l[rgate, i] > -self.wimax_thresh and \
                        self.v_l[rgate, i] > -self.vimax_thresh:
                    self.qflg[rgate,i] = 1
                
                if self.p_l[rgate,i] > self.pwr_keepthresh:
                    self.keep[rgate, i] = 1
                
        # scale p_l by 10 * log10 to match fitacf
        self.p_l = 10 * np.log10(self.p_l)

        # TODO: also calculate "sigma" parameters (call iterative_bayes again with model = 2)
                           
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

    # calculate and store bad lags
    def CalcBadlags(self):
        bad_lags = [[] for i in range(self.nranges)]
        
        # get bad lags - transmit pulse overlap
        for i in range(self.nranges):
            bad_lags[i] = get_badlags(self.rawacf, i)
        
        # get bad lags - power exceeds lag zero power
        # "Spectral width of SuperDARN echos", Ponomarenko and Waters
        for rgate in self.ranges:
            # .. this will only work if we have a good lag zero sample
            # TODO: work on fall back
            if not bad_lags[rgate][0]: 
                offset = self.nlags * rgate
                i_lags = np.array(self.acfi[offset:offset+self.nlags])
                q_lags = np.array(self.acfq[offset:offset+self.nlags])
                samples = i_lags + 1j * q_lags 
                
                lagpowers = abs(samples) ** 2
                bad_lags[rgate] += (lagpowers > lagpowers[0])# add interference lags
            else:
                # if lag zero is bad, we can't filter out lags greater than lag zero because we don't know what it is..
                pass 

        self.bad_lags = bad_lags 

def PlotMixed(lomb):
    ionospheric_scatter = np.sum(lomb.iflg * lomb.qflg, axis=1) > 0
    surface_scatter = np.sum(lomb.gflg * lomb.qflg, axis=1) > 0
    mixed_scatter = (ionospheric_scatter * surface_scatter).nonzero()[0]
    if len(mixed_scatter):
        for mixed in mixed_scatter:
            lomb.PlotPeak(mixed)

# create a COMPACT type h5py dataset using low level API...
def add_compact_dset(hdf5file, group, dsetname, data, dtype, mask = []):
    dsetname = (group + '/' + dsetname).encode()
    if mask != []:
        # save entire row if good data
        mask = np.array([sum(l) for l in mask]) > 0
        data = data[mask]

    dims = data.shape
    space_id = h5py.h5s.create_simple(dims)
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_layout(h5py.h5d.COMPACT)

    dset = h5py.h5d.create(hdf5file.id, dsetname, dtype, space_id, dcpl)
    dset.write(h5py.h5s.ALL, h5py.h5s.ALL, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processes RawACF files with a Lomb-Scargle periodogram to produce FitACF-like science data.')
    
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--infile", help="input RawACF file to convert")
    parser.add_argument("--outfile", help="output FitLSS file")
    parser.add_argument("--starttime", help="input RawACF file to convert")
    parser.add_argument("--endtime", help="input RawACF file to convert")
    
    args = parser.parse_args() 

    # good time at McM is 3/20/2013, 8 AM UTC
    #infile = '/mnt/windata/sddata/0207/all.rawacf'
    # todo: add converging fits
    dfile = DMapFile(files=[args.infile])

    if not args.outfile:
        outfilename  = args.infile.split('/')[-1].rstrip('.bz2').rstrip('rawacf').rstrip('.') + '.fitlomb.hdf5'

    else:
        outfilename = args.outfile

    times = dfile.times
    cubecache = TimeCube()
    print DATA_DIR + outfilename
    hdf5file = h5py.File(DATA_DIR + outfilename, 'w')

    lombfits = []
    for (i,t) in enumerate(times):
        #if(dfile[t]['bmnum'] != 9):
        #    continue
        #if t < datetime.datetime(2014, 3, 24, 17, 30):
        #    continue
        #if t > datetime.datetime(2014, 3, 24, 17, 55):
        #    break
        print 'processing time ' + str(t)
        fit = LombFit(dfile[t])
        
        #fit.ParallelProcessPulse()
        # alternately, use non-parallelized version (easier to debug/optimize)
        fit.ProcessPulse(cubecache)

        fit.WriteLSSFit(hdf5file)

    hdf5file.close() 
    
    del dfile
    
