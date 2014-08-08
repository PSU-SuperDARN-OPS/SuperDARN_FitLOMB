# jon klein, jtklein@alaska.
# functions to calculate a fitlomb (generalized lomb-scargle peridogram) from a rawacf
# parallized with python pp, possible to parallelize over the network
# mit license

# TODO: move raw data to ARSC, process on their machines
# TODO: look at zooming off fit snr for each iteration
# TODO: look at residual spread of fitacf and fitlomb to samples
# TODO: look at variance of residual, compare with fitacf
# TODO: add generalized timecube generation

import argparse
import pydarn.radar as radar
import pydarn.sdio as sdio

from timecube import TimeCube, make_spacecube

import datetime, calendar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import h5py
import lagstate
import pp
import pdb
import os

from iterative_bayes import iterative_bayes, find_fwhm, calculate_bayes, calc_zoomvar

FITLOMB_REVISION_MAJOR = 1
FITLOMB_REVISION_MINOR = 2
ORIGIN_CODE = 'pydarn_fitlombgen.py'
DATA_DIR = '/mnt/flash/sddata/fitlomb/'
FITLOMB_README = 'This group contains data from one SuperDARN pulse sequence with Lomb-Scargle Periodogram fitting.'

I_OFFSET = 0
Q_OFFSET = 1

FWHM_TO_SIGMA = 2.355 # conversion of fwhm to std deviation, assuming gaussian
MAX_V = 1500 # m/s, max velocity (doppler shift) to include in lomb
MAX_W = 1200 # m/s, max spectral width to include in lomb 
SNR_THRESH = 1 # ratio of power in fitted signal and residual 
C = 3e8

CALC_SIGMA = True 

GROUP_ATTR_TYPES = {\
        'txpow':np.int16,\
        'nave':np.int16,\
        'atten':np.int16,\
        'lagfr':np.int16,\
        'smsep':np.int16,\
        'ercod':np.int16,\
        'stat.agc':np.int16,\
        'stat.lopwr':np.int16,\
        'noise.search':np.float32,\
        'noisesky':np.float32,\
        'noisesearch':np.float32,\
        'noise.mean':np.float32,\
        'noisemean':np.float32,\
        'channel':np.int16,\
        'bmnum':np.int16,\
        'bmazm':np.float32,\
        'scan':np.int16,\
        'offset':np.int16,\
        'rxrise':np.int16,\
        'tfreq':np.int16,\
        'mxpwr':np.int32,\
        'lvmax':np.int32,\
        'combf':str,\
        'intt.sc':np.int16,\
        'inttsc':np.int16,\
        'intt.us':np.int32,\
        'inttus':np.int32,\
        'txpl':np.int16,\
        'mpinc':np.int16,\
        'mppul':np.int16,\
        'mplgs':np.int16,\
        'mplgexs':np.int16,\
        'nrang':np.int16,\
        'frang':np.int16,\
        'rsep':np.int16,\
        'ptab':np.int16,\
        'ltab':np.int16,\
        'ifmode':np.int16,\
        'xcf':np.int8}

class LombFit:
    def __init__(self, record):
        self.rawacf = record # dictionary copy of RawACF record
        self.mplgs = self.rawacf.prm.mplgs # range of lags
        self.ranges = range(self.rawacf.prm.nrang) # range gates
        self.nrang = self.rawacf.prm.nrang # range gates
        self.ptab = self.rawacf.prm.ptab # (mppul length list): pulse table
        self.ltab = self.rawacf.prm.ltab # (mplgs x 2 length list): lag table
        self.lagfr = self.rawacf.prm.lagfr # lag to first range in us
        self.mpinc = self.rawacf.prm.mpinc # multi pulse increment (tau, basic lag time) 
        self.txpl = self.rawacf.prm.txpl # 
        self.mppul = self.rawacf.prm.mppul # 
        self.smsep = self.rawacf.prm.smsep 
        acfd = np.array(record.rawacf.acfd)
        self.acfi = acfd[:,:,I_OFFSET]
        self.acfq = acfd[:,:,Q_OFFSET]
        self.tfreq = self.rawacf.prm.tfreq # transmit frequency (kHz)
        self.bmnum = self.rawacf.bmnum # beam number
        self.pwr0 = self.rawacf.fit.pwr0 # pwr0
        self.recordtime = record.time 
        
        # calculate max decay rate for MAX_W spectral width
        amax = (np.pi * 2 * self.tfreq * 1e3 * MAX_W) / C 

        # calculate max frequency either nyquist rate, or calculated off max velocity
        # stepping of alpha on first pass is limited to even integers
        # stepping of frequency on first pass is limited to even integers 
        nyquist = 1 / (2e-6 * self.mpinc)
        fmax = np.ceil(min((MAX_V * 2 * (self.tfreq * 1e3)) / C, nyquist))
        df = 6  #(VEL_RES * 2.) * (self.tfreq * 1e3) / C
        fmax += fmax % df

        self.freqs = np.arange(-fmax,fmax, df)
        
        self.maxalf = amax
        da = 6
        self.maxfreqs = 1
        self.alfs = np.arange(0, self.maxalf, da)
        
        # thresholds on velocity and spectral width for surface scatter flag (m/s)
        self.v_thresh = 30.
        self.w_thresh = 90. # blanchard, 2009
        
        # threshold on power (snr), spectral width std error m/s, and velocity std error m/s for quality flag
        self.qwle_thresh = 80
        self.qvle_thresh = 80
        self.qpwr_thresh = 2
        self.snr_thresh = SNR_THRESH 
        # thresholds on velocity and spectral width for ionospheric scatter flag (m/s)
        self.wimin_thresh = 100
        self.wimax_thresh = MAX_W - 100
        self.vimax_thresh = MAX_V - 100
        self.vimin_thresh = 100
       
        # initialize empty arrays for fitted parameters 
        self.lfits      = [[] for r in self.ranges]
        self.sfits      = [[] for r in self.ranges]

        self.sd_s       = np.zeros([self.nrang, self.maxfreqs])
        self.w_s_e      = np.zeros([self.nrang, self.maxfreqs])
        self.w_s_std    = np.zeros([self.nrang, self.maxfreqs])
        self.w_s        = np.zeros([self.nrang, self.maxfreqs])
        self.p_s        = np.zeros([self.nrang, self.maxfreqs])
        self.p_s_e      = np.zeros([self.nrang, self.maxfreqs])
        self.v_s        = np.zeros([self.nrang, self.maxfreqs])
        self.v_s_e      = np.zeros([self.nrang, self.maxfreqs])
        self.v_s_std    = np.zeros([self.nrang, self.maxfreqs])

        self.w_l_e      = np.zeros([self.nrang, self.maxfreqs])
        self.w_l_std    = np.zeros([self.nrang, self.maxfreqs])
        self.w_l        = np.zeros([self.nrang, self.maxfreqs])

        self.fit_snr_l  = np.zeros([self.nrang, self.maxfreqs])
        self.fit_snr_s  = np.zeros([self.nrang, self.maxfreqs])

        self.r2_phase_l  = np.zeros([self.nrang, self.maxfreqs])
        self.r2_phase_s  = np.zeros([self.nrang, self.maxfreqs])

        self.p_l        = np.zeros([self.nrang, self.maxfreqs])
        self.p_l_e      = np.zeros([self.nrang, self.maxfreqs])
        self.v_l        = np.zeros([self.nrang, self.maxfreqs])
        self.v_l_e      = np.zeros([self.nrang, self.maxfreqs])
        self.v_l_std    = np.zeros([self.nrang, self.maxfreqs])

        self.gflg       = np.zeros([self.nrang, self.maxfreqs])
        self.iflg       = np.zeros([self.nrang, self.maxfreqs])
        self.qflg       = np.zeros([self.nrang, self.maxfreqs])

        self.nlag       = np.zeros([self.nrang])
        
        self.CalcBadlags()
        self.CalcNoise()
 
    # appends a record of the lss fit to an hdf5 file
    def WriteLSSFit(self, hdf5file):
        groupname = str(calendar.timegm(self.recordtime.timetuple()))
        grp = hdf5file.create_group(groupname)
        # add scalars as attributes to group
        for attr in self.rawacf.prm.__dict__.keys():
            if self.rawacf.prm.__dict__[attr] != None:
                grp.attrs[attr] = GROUP_ATTR_TYPES[attr](self.rawacf.prm.__dict__[attr])

        # add scalars with changed names on davitpy..
        grp.attrs['noise.search'] = np.float32(self.rawacf.prm.noisesearch)
        grp.attrs['noise.mean'] = np.float32(self.rawacf.prm.noisemean)
        grp.attrs['intt.sc'] = np.int16(self.rawacf.prm.inttsc)
        grp.attrs['intt.us'] = np.int32(self.rawacf.prm.inttus)
        grp.attrs['channel'] = np.int16(ord(self.rawacf.channel) - ord('a'))
        grp.attrs['bmnum'] = np.int16(self.rawacf.bmnum)

        # add times..
        grp.attrs['time.yr'] = np.int16(self.recordtime.year)
        grp.attrs['time.mo'] = np.int16(self.recordtime.month) 
        grp.attrs['time.dy'] = np.int16(self.recordtime.day)
        grp.attrs['time.hr'] = np.int16(self.recordtime.hour) 
        grp.attrs['time.mt'] = np.int16(self.recordtime.minute)
        grp.attrs['time.sc'] = np.int16(self.recordtime.second)
        grp.attrs['time.us'] = np.int32(self.recordtime.microsecond) 

        grp.attrs['readme'] = FITLOMB_README
        grp.attrs['fitlomb.revision.major'] = np.int8(FITLOMB_REVISION_MAJOR)
        grp.attrs['fitlomb.revision.minor'] = np.int8(FITLOMB_REVISION_MINOR)
        grp.attrs['fitlomb.bayes.iterations'] = np.int16(self.maxfreqs)
        grp.attrs['origin.code'] = ORIGIN_CODE # TODO: ADD ARGUEMENTS
        grp.attrs['origin.time'] = str(datetime.datetime.now())
        
        grp.attrs['stid'] = np.int16(self.rawacf.stid)
        grp.attrs['cp'] = np.int16(self.rawacf.cp)
        
        grp.attrs['epoch.time'] = calendar.timegm(self.recordtime.timetuple())
        grp.attrs['noise.lag0'] = np.float64(self.noise) # lag zero power from noise acf?
        
        # copy over vectors from rawacf
        add_compact_dset(hdf5file, groupname, 'ptab', np.int16(self.ptab), h5py.h5t.STD_I16BE)
        add_compact_dset(hdf5file, groupname, 'ltab', np.int16(self.ltab), h5py.h5t.STD_I16BE)
        add_compact_dset(hdf5file, groupname, 'pwr0', np.int32(self.pwr0), h5py.h5t.STD_I32BE)
        
        # add calculated parameters
        add_compact_dset(hdf5file, groupname, 'qflg', np.int32(self.qflg), h5py.h5t.STD_I32BE)
        add_compact_dset(hdf5file, groupname, 'gflg', np.int8(self.gflg), h5py.h5t.STD_I8BE)
        add_compact_dset(hdf5file, groupname, 'iflg', np.int8(self.iflg), h5py.h5t.STD_I8BE)
        add_compact_dset(hdf5file, groupname, 'nlag', np.int16(self.nlag), h5py.h5t.STD_I16BE)
        
        add_compact_dset(hdf5file, groupname, 'p_l', np.float64(self.p_l), h5py.h5t.NATIVE_DOUBLE)
        add_compact_dset(hdf5file, groupname, 'p_l_e', np.float64(self.p_l_e), h5py.h5t.NATIVE_DOUBLE)
        add_compact_dset(hdf5file, groupname, 'w_l', np.float64(self.w_l), h5py.h5t.NATIVE_DOUBLE)
        add_compact_dset(hdf5file, groupname, 'w_l_e', np.float64(self.w_l_e), h5py.h5t.NATIVE_DOUBLE)
        add_compact_dset(hdf5file, groupname, 'w_l_std', np.float64(self.w_l_std), h5py.h5t.NATIVE_DOUBLE)
        add_compact_dset(hdf5file, groupname, 'v', np.float64(self.v_l), h5py.h5t.NATIVE_DOUBLE)
        add_compact_dset(hdf5file, groupname, 'v_e', np.float64(self.v_l_e), h5py.h5t.NATIVE_DOUBLE)
        add_compact_dset(hdf5file, groupname, 'v_l_std', np.float64(self.v_l_std), h5py.h5t.NATIVE_DOUBLE)
        add_compact_dset(hdf5file, groupname, 'fit_snr_l', np.float64(self.fit_snr_l), h5py.h5t.NATIVE_DOUBLE)
        add_compact_dset(hdf5file, groupname, 'r2_phase_l', np.float64(self.r2_phase_l), h5py.h5t.NATIVE_DOUBLE)

        if CALC_SIGMA:
            add_compact_dset(hdf5file, groupname, 'p_s', np.float64(self.p_s), h5py.h5t.NATIVE_DOUBLE)
            add_compact_dset(hdf5file, groupname, 'p_s_e', np.float64(self.p_s_e), h5py.h5t.NATIVE_DOUBLE)
            add_compact_dset(hdf5file, groupname, 'w_s', np.float64(self.w_s), h5py.h5t.NATIVE_DOUBLE)
            add_compact_dset(hdf5file, groupname, 'w_s_e', np.float64(self.w_s_e), h5py.h5t.NATIVE_DOUBLE)
            add_compact_dset(hdf5file, groupname, 'w_s_std', np.float64(self.w_s_std), h5py.h5t.NATIVE_DOUBLE)
            add_compact_dset(hdf5file, groupname, 'v_s', np.float64(self.v_s), h5py.h5t.NATIVE_DOUBLE)
            add_compact_dset(hdf5file, groupname, 'v_s_e', np.float64(self.v_s_e), h5py.h5t.NATIVE_DOUBLE)
            add_compact_dset(hdf5file, groupname, 'v_s_std', np.float64(self.v_s_std), h5py.h5t.NATIVE_DOUBLE)
            add_compact_dset(hdf5file, groupname, 'fit_snr_s', np.float64(self.fit_snr_s), h5py.h5t.NATIVE_DOUBLE)
            add_compact_dset(hdf5file, groupname, 'r2_phase_s', np.float64(self.r2_phase_s), h5py.h5t.NATIVE_DOUBLE)
        
    # calculates FitACF-like parameters for each peak in the spectrum
    # processes the pulse (move it __init__)?
    def ProcessPulse(self, cubecache):
        for r in self.ranges:
            peaks = self.CalculatePeaks(r, cubecache)
            if CALC_SIGMA:
                peaks = self.CalculatePeaks(r, cubecache, env_model = 2)
        self.ProcessPeaks()
    
    def ProcessMultiPulse(self, cubecache, otherpulses):
        for r in self.ranges:
            peaks = self.CalculatePeaks(r, cubecache, otherpulses = otherpulses)
            if CALC_SIGMA:
                peaks = self.CalculatePeaks(r, cubecache, env_model = 2, otherpulses = otherpulses)
        self.ProcessPeaks()


    # do comparison plots of the lomb fit and fitacf in lagspace
    # todo: source data with pydarn sdio..
    def FitACFCompare(self, fitacfs, radar, plot_lagspace = True):
        # look at residual spread of fitacf and fitlomb to samples
        # look at variance of residual, compare with 

        w_fitacf = []
        w_lomb = []

        v_fitacf = []
        v_lomb = []

        p_fitacf = []
        p_lomb = []

        fitacf = fitacfs[self.recordtime]

        w_fitacf.append(np.array(fitacf['w_l']))
        w_lomb = self.w_l 

        v_fitacf.append(np.array(fitacf['v']))
        v_lomb = self.v_l

        p_fitacf.append(np.array(fitacf['p_l']))
        p_lomb = self.p_l

        for (i,s) in enumerate(fitacf['slist']):
            # scatter plot samples, subplot, make stuff bigger.
            # only compare quality fits?
            fit = self.lfits[s][0]
            # plot fitacf and fitlomb fits
            samples = fit['samples']

            tsamp = fit['t']
            t = np.arange(0, .04, .0015)

            fitsignal =  fit['amplitude'] * np.exp(1j * 2 * np.pi * fit['frequency'] * t) * np.exp(-fit['alpha'] * t)
            noise = fitacf['noise.sky']
            try:
                amp = noise * (10 ** (fitacf['p_l'][i]/10))
            except:
                pdb.set_trace()
            f = (fitacf['v'][i] * 2 * self.tfreq * 1e3) / C

            alf = (fitacf['w_l'][i] * 2 * np.pi * fitlomb.tfreq * 1e3 * fitlomb.t_pulse * 1e-6) / C 
            facf = amp * np.exp(1j * 2 * np.pi * f * t) * np.exp(-alf * t)
            print 'gate : ' + str(s)
            print 'fitacf alpha: ' + str(alf)
            print 'lomb alpha  : ' + str(fit['alpha'])
            print ''
            if fit['alpha'] < 0:
                pdb.set_trace()
    
            if plot_lagspace:
                plt.rcParams.update({'legend.loc': 'best'})
                plt.legend(fancybox=True)
                plt.rcParams.update({'font.size': 12})
                ymax = max(max(np.real(samples)),max(np.imag(samples))) * 3 / 1000
                ymin = -ymax
                xmin = min(t)
                xmax = max(t)

                plt.subplot(2,1,1)
                plt.scatter(tsamp,np.real(samples)/1000, s=20, c='r')
                plt.plot(t,np.real(fitsignal)/1000, c='b')
                plt.plot(t,np.real(facf)/1000, c ='m')
                plt.legend(['fitlomb', 'fitacf', 'q samples'])
                plt.ylabel('sample value')
                plt.title('Comparison of FitACF and FitLOMB in lagspace\n' + radar + ' beam: ' + str(self.bmnum) + ' range: ' + str(s) + ', ' + self.rawacf['origin.time'])
                plt.axis([xmin, xmax, ymin, ymax])            

                plt.subplot(2,1,2)
                plt.scatter(tsamp,np.imag(samples)/1000, s=20, c='r')
                plt.plot(t,np.imag(fitsignal)/1000, c='b')
                plt.plot(t,np.imag(facf)/1000, c='m')
                plt.legend(['fitlomb', 'fitacf', 'i samples'])
                plt.ylabel('sample value')
                plt.xlabel('lag time (seconds)')

                plt.axis([xmin, xmax, ymin, ymax])            
                imgname = 'lagplots/' + radar + '_' + str(s) + '_' + str(calendar.timegm(self.recordtime.timetuple()) + int(self.rawacf['time.us'])/1e6) + '.png'
                plt.savefig(imgname, bbox_inches='tight')
                plt.clf()

    # get time and good complex samples for a range gate
    def _CalcSamples(self, rgate):
        offset = self.mplgs * rgate

        # see http://davit.ece.vt.edu/davitpy/_modules/pydarn/sdio/radDataTypes.html
        i_lags = self.acfi[rgate]
        q_lags = self.acfq[rgate]
        
        good_lags = np.ones(self.mplgs)
        good_lags[self.bad_lags[rgate] != 0] = 0

        lags = map(lambda x : abs(x[1]-x[0]), self.ltab[0:self.mplgs])
        
        i_lags = i_lags[good_lags == True]
        q_lags = q_lags[good_lags == True]

        t = (np.array(lags) * self.mpinc / 1e6)[good_lags == True]
        samples = i_lags + 1j * q_lags
        return t, samples


    # finds returns in spectrum and records cell velocity
    def CalculatePeaks(self, rgate, cubecache, env_model = 1, otherpulses = None):
        t, samples = self._CalcSamples(rgate)
        
        if otherpulses:
            for op in otherpulses:
                ot, osamples = op._CalcSamples(rgate)
                t = np.concatenate([t, ot])
                samples = np.concatenate([samples, osamples])
        
        # calcuate generalized lomb-scargle periodogram iteratively
        if env_model == 1:
            self.lfits[rgate] = iterative_bayes(samples, t, self.freqs, self.alfs, env_model, self.maxfreqs, cubecache = cubecache)
        elif env_model == 2:
            self.sfits[rgate] = iterative_bayes(samples, t, self.freqs, self.alfs, env_model, self.maxfreqs, cubecache = cubecache)

    def ProcessPeaks(self):
        # compute velocity, width, and power for each peak with "sigma" and "lambda" fits
        for rgate in self.ranges:
            for (i, fit) in enumerate(self.lfits[rgate]):
                N = 2 * len(fit['t'])  

                # calculate "lambda" parameters
                # see Effects of mixed scatter on SuperDARN convection maps near (1) for spectral width 
                # see ros.3.6/codebase/superdarn/src.lib/tk/fitacf/src/fit_acf.c and do_fit.c
                self.w_l[rgate,i] = (C * fit['alpha']) / (2. * np.pi * (self.tfreq * 1e3)) 
                # approximate alpha error by taking half of range of alphas covered in fwhm
                # results in standard deviation
                self.w_l_std[rgate,i] = (((C * fit['alpha_fwhm']) / (2. * np.pi * (self.tfreq * 1e3))) / FWHM_TO_SIGMA)
                self.w_l_e[rgate,i] = self.w_l_std[rgate,i] / np.sqrt(N)
                
                # record ratio of power in signal versus power in fitted signal
                self.fit_snr_l[rgate,i] = fit['fit_snr']

                # amplitude estimation, see bayesian analysis v: amplitude estimation, multiple well separated sinusoids
                # bretthorst, equation 78, I'm probably doing this wrong...
                # to match fitacf, scale p_l by 10 * log10
                self.p_l[rgate,i] = fit['amplitude'] / self.noise
                self.p_l_e[rgate,i] = np.sqrt((self.noise ** 2)/fit['amplitude_error_unscaled'])/self.noise # this may be scaled wrong..

                v_l = (fit['frequency'] * C) / (2 * self.tfreq * 1e3)
                self.v_l[rgate,i] = v_l
                # approximate velocity error as half the range of velocities covered by fwhm 
                # "nonuniform sampling: bandwidth and aliasing", page 25

                self.v_l_std[rgate,i] = ((((fit['frequency_fwhm']) * C) / (2 * self.tfreq * 1e3)) / FWHM_TO_SIGMA)
                self.v_l_e[rgate,i] = self.v_l_std[rgate,i] / np.sqrt(N)
                
                self.r2_phase_l[rgate, i]  = fit['r2_phase']
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
                        self.fit_snr_l[rgate, i] > self.snr_thresh and \
                        self.v_l[rgate, i] > -self.vimax_thresh:
                    self.qflg[rgate,i] = 1
                
            if CALC_SIGMA:
                for (i, fit) in enumerate(self.sfits[rgate]):
                    # calculate "sigma" parameters
                    np.append(self.sd_s[rgate],0) # TODO: what is a reasonable value for this? 
                     
                    self.w_s[rgate,i] = (C * fit['alpha']) / (2. * np.pi * (self.tfreq * 1e3)) 

                    self.w_s_std[rgate,i] = (((C * fit['alpha_fwhm']) / (2. * np.pi * (self.tfreq * 1e3))) / FWHM_TO_SIGMA)
                    self.w_s_e[rgate,i] = self.w_s_std[rgate,i] / np.sqrt(N)
                    
                    self.p_s[rgate,i] = fit['amplitude'] / self.noise
                    self.p_s_e[rgate,i] = np.sqrt((self.noise ** 2)/fit['amplitude_error_unscaled'])/self.noise # this may be scaled wrong..

                    v_s = (fit['frequency'] * C) / (2 * self.tfreq * 1e3)
                    self.v_s[rgate,i] = v_s

                    self.r2_phase_s[rgate, i] = fit['r2_phase']
                    self.fit_snr_s[rgate,i] = fit['fit_snr']
                    self.v_s_std[rgate,i] = ((((fit['frequency_fwhm']) * C) / (2 * self.tfreq * 1e3)) / FWHM_TO_SIGMA)
                    self.v_s_e[rgate,i] = self.v_s_std[rgate,i] / np.sqrt(N)
                self.p_s[self.p_s <= 0] = np.nan 
                self.p_s = 10 * np.log10(self.p_s)

        # scale p_l by 10 * log10 to match fitacf
        self.p_l[self.p_l <= 0] = np.nan 
        self.p_l = 10 * np.log10(self.p_l)

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

    def CalcNoise(self):
        # take average of smallest ten powers at range gate 0 for lower bound on noise
        pnmin = np.mean(sorted(self.pwr0)[:10])
        self.noise = pnmin

        # take 1.6 * pnmin as upper bound for noise, 
        pnmax = 1.6 * pnmin # why 1.6? because fitacf does it that way...
        
        noise_samples = np.array([])

        # look through good lags for ranges with pnmin, pnmax for more noise samples
        noise_ranges = (self.pwr0 > pnmin) * (self.pwr0 < pnmax)
        
        for r in np.nonzero(noise_ranges)[0]:
            t, samples = self._CalcSamples(r)
            
            noise_lags = np.nonzero((abs(samples) > pnmin) * (abs(samples) < pnmax))[0]
            noise_samples = np.append(noise_samples, abs(samples)[noise_lags])
       
        # set noise as average of noise samples between pnmin and pnmax
        if len(noise_samples):
            self.noise = np.mean(noise_samples)
    
    # calculate and store bad lags
    def CalcBadlags(self, pwrthresh = True):
        bad_lags = lagstate.bad_lags(self, self.pwr0)
      
        if pwrthresh:
            # get bad lags - power exceeds lag zero power
            # "Spectral width of SuperDARN echos", Ponomarenko and Waters
            for rgate in self.ranges:
                # .. this will only work if we have a good lag zero sample
                # TODO: work on fall back
                if not bad_lags[rgate][0]: 
                    offset = self.mplgs * rgate
                    i_lags = self.acfi[rgate]
                    q_lags = self.acfq[rgate]
                    samples = i_lags + 1j * q_lags 
                    
                    lagpowers = abs(samples) ** 2

                    bad_lags[rgate] += (lagpowers > (lagpowers[0] * 2.0))# add interference lags
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
    
    parser.add_argument("--infile", help="input RawACF file to convert")
    parser.add_argument("--outfile", help="output FitLSS file")
    parser.add_argument("--starttime", help="input RawACF file to convert")
    parser.add_argument("--endtime", help="input RawACF file to convert")
    parser.add_argument("--pulses", help="calculate lomb over multiple pulses", default=1) 
    parser.add_argument("--radar", help="radar to create data from", default='kod.c') 

    args = parser.parse_args() 
    args.pulses = int(args.pulses)

    # ksr, kod, pgr, ade, cvw 
    recordlen = 2
    days = [datetime.datetime(2014,2,27)]#datetime.datetime(2014,2,27), datetime.datetime(2014,3,1), datetime.datetime(2014,3,2), datetime.datetime(2014,3,4), datetime.datetime(2014,3,6)]
    hours = range(0, 24, recordlen) 
    #hours = [0]
    #recordlen = .1
    #stimes = [datetime.datetime(2014,2,26,1,20)]
    #etimes = [datetime.datetime(2014,2,26,1,22)]

    # boilerplate..
    fileType='rawacf'
    filtered=False
    src='local'
    channel=None

    for radar_code in [args.radar]:
        for sday in days:
            for hoffset in hours:
                stime = sday + datetime.timedelta(hours = hoffset)
                etime = sday + datetime.timedelta(hours = (hoffset + recordlen))
                myPtr = sdio.radDataOpen(stime,radar_code,eTime=etime,channel=None,bmnum=None,cp=None,fileType=fileType,filtered=filtered, src=src)
                outfilename = stime.strftime('%Y%m%d.%H%M.' + radar_code + '.fitlomb.hdf5') 
                outfilepath = DATA_DIR + stime.strftime('%Y/%m.%d/') 
                if not os.path.exists(outfilepath):
                    os.makedirs(outfilepath)
                cubecache = TimeCube()
                
                hdf5file = h5py.File(outfilepath + outfilename, 'w')

                lombfits = []
                 
                drec = sdio.radDataReadRec(myPtr)

                while drec != None:
                    fit = LombFit(drec)
                    fit.ProcessPulse(cubecache)
                    fit.WriteLSSFit(hdf5file)
                    drec = sdio.radDataReadRec(myPtr)

                hdf5file.close() 
