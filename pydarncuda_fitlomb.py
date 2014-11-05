# jon klein, jtklein@alaska.
# functions to calculate a fitlomb (generalized lomb-scargle peridogram) from a rawacf
# mit license

# TODO: move raw data to ARSC, process on their machines
# TODO: look at residual spread of fitacf and fitlomb to samples
# TODO: look at variance of residual, compare with fitacf
# TODO: calculate bad lags off of pulse sequence only once per pulse sequence for ~40% speedup
# TODO: store data in hdf5 file with large vector for entire record, rather than datasets for each?
# (should reduce data usage and allow for compression)
import argparse
import pydarn.sdio as sdio
import datetime, calendar, time
import numpy as np
import h5py
import lagstate
import pdb
import os
import getpass

from multiprocessing import Pool, Manager 
from bigdipper import cache_data, mount_raid0

FITLOMB_REVISION_MAJOR = 3
FITLOMB_REVISION_MINOR = 0
ORIGIN_CODE = 'pydarncuda_fitlomb.py'
DATA_DIR = '/home/' + getpass.getuser() + '/fitlomb/'
FITLOMB_README = 'This group contains data from one SuperDARN pulse sequence with Lomb-Scargle Periodogram fitting.'

I_OFFSET = 0
Q_OFFSET = 1

FWHM_TO_SIGMA = 2.355 # conversion of fwhm to std deviation, assuming gaussian
MAX_V = 2000 # m/s, max velocity (doppler shift) to include in lomb
MAX_W = 2000 # m/s, max spectral width to include in lomb 
NFREQS = 128 
NALFS = 128 
MAXPULSES = 300
LAMBDA_FIT = 1
SIGMA_FIT = 2
SNR_THRESH = .25 # minimum ratio of power in fitted signal and residual for a qualtiy fit
VERR_THRESH = 50 
WERR_THRESH = 50 
C = 3e8
MAX_TFREQ = 16e6
POOL_SIZE = 8 # maximum pool size for multiprocessing of fits..
CALC_SIGMA = False 
DEBUG = False

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

class CULombFit:
    #@profile
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
               
        # thresholds on velocity and spectral width for surface scatter flag (m/s)
        self.v_thresh = 30.
        self.w_thresh = 90. # blanchard, 2009
        
        # threshold on power (snr), spectral width std error m/s, and velocity std error m/s for quality flag
        self.qwle_thresh = WERR_THRESH
        self.qvle_thresh = VERR_THRESH 
        self.qpwr_thresh = .5
        self.snr_thresh = SNR_THRESH 
        # thresholds on velocity and spectral width for ionospheric scatter flag (m/s)
        self.wimin_thresh = 100
        self.wimax_thresh = MAX_W - 50
        self.vimax_thresh = MAX_V - 50
        self.vimin_thresh = 100
        
        self.maxfreqs = 1
        # initialize empty arrays for fitted parameters 

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

        self.nlag       = np.zeros([self.nrang, self.maxfreqs])

        self.CalcLags()
        self.CalcBadlags() # TODO: about 1/3 of execution time spent here 
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
        #add_compact_dset(hdf5file, groupname, 'gflg', np.int8(self.gflg), h5py.h5t.STD_I8BE)
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
        #add_compact_dset(hdf5file, groupname, 'r2_phase_l', np.float64(self.r2_phase_l), h5py.h5t.NATIVE_DOUBLE)

        if CALC_SIGMA:
            add_compact_dset(hdf5file, groupname, 'p_s', np.float64(self.p_s), h5py.h5t.NATIVE_DOUBLE)
            #add_compact_dset(hdf5file, groupname, 'p_s_e', np.float64(self.p_s_e), h5py.h5t.NATIVE_DOUBLE)
            add_compact_dset(hdf5file, groupname, 'w_s', np.float64(self.w_s), h5py.h5t.NATIVE_DOUBLE)
            add_compact_dset(hdf5file, groupname, 'w_s_e', np.float64(self.w_s_e), h5py.h5t.NATIVE_DOUBLE)
            add_compact_dset(hdf5file, groupname, 'w_s_std', np.float64(self.w_s_std), h5py.h5t.NATIVE_DOUBLE)
            add_compact_dset(hdf5file, groupname, 'v_s', np.float64(self.v_s), h5py.h5t.NATIVE_DOUBLE)
            add_compact_dset(hdf5file, groupname, 'v_s_e', np.float64(self.v_s_e), h5py.h5t.NATIVE_DOUBLE)
            add_compact_dset(hdf5file, groupname, 'v_s_std', np.float64(self.v_s_std), h5py.h5t.NATIVE_DOUBLE)
            add_compact_dset(hdf5file, groupname, 'fit_snr_s', np.float64(self.fit_snr_s), h5py.h5t.NATIVE_DOUBLE)
            #add_compact_dset(hdf5file, groupname, 'r2_phase_s', np.float64(self.r2_phase_s), h5py.h5t.NATIVE_DOUBLE)
    #@profile 
    def CudaProcessPulse(self, gpu):
        lagsmask = []
        isamples = np.zeros([len(self.ranges), 2 * gpu.nlags])

        # about 15% of execution time spent here
        for r in self.ranges:
            times, samples = self._CalcSamples(r)
            lmask = [l in times for l in gpu.lags]
            lagsmask.append(lmask)
            
            # create interleaved samples array (todo: don't calculate bad samples for ~2x speedup)
            i = 0
            for (j,l) in enumerate(lmask):
                if l:
                    isamples[r,2*j] = np.real(samples[i])
                    isamples[r,2*j+1] = np.imag(samples[i])
                    i = i + 1

        
        lagsmask = np.int8(np.array(lagsmask))
        self.isamples = np.float32(np.array(isamples))

        gpu.run_bayesfit(self.isamples, lagsmask) # TODO: calculate only on good lags for ~50% speedup?
        gpu.process_bayesfit(self.tfreq, self.noise)


    # get time and good complex samples for a range gate
    def _CalcSamples(self, rgate):
        # see http://davit.ece.vt.edu/davitpy/_modules/pydarn/sdio/radDataTypes.html
        i_lags = self.acfi[rgate]
        q_lags = self.acfq[rgate]
        
        good_lags = np.ones(self.mplgs)
        good_lags[self.bad_lags[rgate] != 0] = 0

        i_lags = i_lags[good_lags == True]
        q_lags = q_lags[good_lags == True]

        t = self.lags[good_lags == True]
        samples = i_lags + 1j * q_lags
        return t, samples

    def CalcLags(self):
        self.lags = np.float32(np.array(map(lambda x : abs(x[1]-x[0]), self.ltab[0:self.mplgs])) * (self.mpinc / 1e6))

    def CudaCopyPeaks(self, gpu, itr = 0):

        if gpu.env_model == LAMBDA_FIT:

            self.w_l[:,itr] = gpu.w 

            self.w_l_std[:,itr] = gpu.w_std
            self.w_l_e[:,itr] = gpu.w_e
 
            self.v_l[:,itr] = gpu.v 
            self.v_l_std[:,itr] = gpu.v_std
            self.v_l_e[:,itr] = gpu.v_e

            self.p_l[:,itr] = gpu.p
            self.fit_snr_l[:,itr] = gpu.snr # record ratio of power in signal versus power in fitted signal
            
            iflg = (abs(self.v_l) - (self.v_thresh - (self.v_thresh / self.w_thresh) * abs(self.w_l)) > 0) 
            self.iflg[:,itr][iflg[:,0]] = 1
            qflg = (self.p_l > self.qpwr_thresh) * \
                   (self.w_l_e < self.qwle_thresh) * \
                   (self.v_l_e < self.qvle_thresh) * \
                   (self.w_l < self.wimax_thresh) * \
                   (self.v_l < self.vimax_thresh) * \
                   (self.w_l > -self.wimax_thresh) * \
                   (self.fit_snr_l >= self.snr_thresh) * \
                   (self.v_l > -self.vimax_thresh)

            self.qflg[:,itr][qflg[:,0]] = 1

        elif gpu.env_model == SIGMA_FIT:
            self.w_s[:, itr] = gpu.w 
            self.w_s_std[:,itr] = gpu.w_std
            self.w_s_e[:,itr] = gpu.w_e
 
            self.v_s[:,itr] = gpu.v 
            self.v_s_std[:,itr] = gpu.v_std
            self.v_s_e[:,itr] = gpu.v_e

            self.p_s[:,itr] = gpu.p
            self.fit_snr_s[:,itr] = gpu.snr
        
        else:
            print 'error - unknown environment model'
    

    def CudaPlotFit(self, gpu):
        import matplotlib.pyplot as plt

        for gate in self.ranges:
            print self.recordtime
            print 'range gate: ' + str(gate)
            print 'calculated amplitude: ' + str(gpu.amplitudes[gate])
            print 'calculated freq: ' + str(gpu.vfreq[gate])
            print 'calculated decay: ' + str(gpu.walf[gate])
            print 'snr: ' + str(gpu.snr[gate])

            fit = gpu.amplitudes[gate] * np.exp(1j * 2 * np.pi * gpu.vfreq[gate] * gpu.lags) * np.exp(-gpu.walf[gate] * gpu.lags)
            plt.plot(np.real(fit), '-')
            plt.plot(np.imag(fit), '-')
            
            signal = self.isamples[gate][I_OFFSET::2] + 1j*self.isamples[gate][Q_OFFSET::2]

            signal[self.bad_lags[gate] != 0] = 0

            plt.plot(np.real(signal))
            plt.plot(np.imag(signal))
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
    #@profile
    def CalcBadlags(self, pwrthresh = True, uselagzero = False):
        bad_lags = lagstate.bad_lags(self, self.pwr0)
      
        if pwrthresh:
            # get bad lags - power exceeds lag zero power
            # "Spectral width of SuperDARN echos", Ponomarenko and Waters
            for rgate in self.ranges:
                # .. this will only work if we have a good lag zero sample
                # TODO: work on fall back
                if not bad_lags[rgate][0]: 
                    i_lags = self.acfi[rgate]
                    q_lags = self.acfq[rgate]
                    samples = i_lags + 1j * q_lags 
                    
                    lagpowers = abs(samples) ** 2

                    bad_lags[rgate] += (lagpowers > (lagpowers[0] * 10.0))# add interference lags
                else:
                    # if lag zero is bad, we can't filter out lags greater than lag zero because we don't know what it is..
                    pass 
                
                self.nlag[rgate,0] = len(bad_lags[rgate]) - sum(bad_lags[rgate])

                if not uselagzero:
                    bad_lags[rgate][0] = True

        self.bad_lags = bad_lags 

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

# worker function to fitlomb process a block of time
#@profile
def generate_fitlomb(record):
        from cuda_bayes import BayesGPU

        # unpack record tuple (passing multiple arguements with map is awkward..)
        stime, etime, radar, lock = record

        print 'worker computing from ' + str(stime) + ' to ' + str(etime)
        # lock so multiple processes don't step over eachother unpacking and copying rawacfs to /tmp 
        lock.acquire()
        myPtr = sdio.radDataOpen(stime,radar,eTime=etime,channel=None,bmnum=None,cp=None,fileType='rawacf',filtered=False, src='local')
        lock.release()

        outfilename = stime.strftime('%Y%m%d.%H%M.' + radar + '.fitlomb.hdf5') 
        outfilepath = DATA_DIR + stime.strftime('%Y/%m.%d/') 

        if not os.path.exists(outfilepath):
            os.makedirs(outfilepath)
        
        hdf5file = h5py.File(outfilepath + outfilename, 'w')

        # set up frequency/alpha vectors 
        amax = np.ceil((np.pi * 2 * MAX_TFREQ * MAX_W) / C)
        fmax = np.ceil(MAX_V * 2 * MAX_TFREQ / C)
        freqs = np.linspace(-fmax,fmax, NFREQS)
        alfs = np.linspace(0, amax, NALFS)
        
        try: 
            drec = sdio.radDataReadRec(myPtr)

        except:
            print 'error reading first rawacf record for ' + str(stime) + '... skipping to next record block'
            hdf5file.close() 
            return 

        gpu_lambda = None
        if CALC_SIGMA:
            gpu_sigma = None

        while drec != None:
            try:
                fit = CULombFit(drec) # ~ 30% of the time is spent here
            except:
                print 'error reading rawacf record, skipping'
                continue
            
            # create velocity and spectral width space based on maximum transmit frequency
            if gpu_lambda == None:
                gpu_lambda = BayesGPU(fit.lags, freqs, alfs, fit.nrang, LAMBDA_FIT)
                if CALC_SIGMA:
                    gpu_sigma = BayesGPU(fit.lags, freqs, alfs, fit.nrang, SIGMA_FIT)

            # generate new caches on the GPU for the fit if the pulse sequence has changed 
            elif gpu_lambda.npulses != fit.nrang or (not np.array_equal(fit.lags, gpu_lambda.lags)):
                gpu_lambda = BayesGPU(fit.lags, freqs, alfs, fit.nrang, LAMBDA_FIT)
                if CALC_SIGMA:
                    gpu_sigma = BayesGPU(fit.lags, freqs, alfs, fit.nrang, SIGMA_FIT)
                print 'the pulse sequence has changed'
           
            try:
                fit.CudaProcessPulse(gpu_lambda) # ~ 50%
                if CALC_SIGMA:
                    fit.CudaProcessPulse(gpu_sigma) # ~ 50%
                fit.CudaCopyPeaks(gpu_lambda)
                if CALC_SIGMA:
                    fit.CudaCopyPeaks(gpu_sigma)
                fit.WriteLSSFit(hdf5file) # 4 %

                '''if calendar.timegm(fit.recordtime.timetuple()) == 1393676837:
                    import pdb
                    pdb.set_trace()'''

                #fit.CudaPlotFit(gpu_lambda)
	    except:
		print 'error fitting file, skipping record at ' + str(fit.recordtime) 

            #print 'computed ' + str(fit.recordtime)

            drec = sdio.radDataReadRec(myPtr) # ~ 10% of the time is spent here

        hdf5file.close() 


#@profile
def main():
    parser = argparse.ArgumentParser(description='Processes RawACF files with a Lomb-Scargle periodogram to produce FitACF-like science data.')
    
    parser.add_argument("--starttime", help="start time of fit (yyyy.mm.dd.hhMM) e.g 2014.02.25.0000", default = "2014.03.01.0000")
    parser.add_argument("--endtime", help="ending time of fit (yyyy.mm.dd.hhMM) e.g 2014.03.10.0000", default = "2014.03.01.0200")
    parser.add_argument("--enable_sigmafit", help="enable fitting sigma (p_s/v_s) parameters. this will double runtime and GPU VRAM usage", action='store_true', default=False) 
    parser.add_argument("--recordlen", help="breaks the output into recordlen hour length files (max 24)", default=2) 
    parser.add_argument("--poolsize", help="maximum number of simultaneous subprocesses", default=POOL_SIZE) 
    parser.add_argument("--radars", help="radar(s) to process data on", nargs='+', default=['mcm.a'])
    parser.add_argument("--datadir", help="base directory for .fitlomb files (defaults to /home/radar/fitlomb/)", default='/home/radar/fitlomb/') 

    # TODO: add channel/beam?
    args = parser.parse_args() 
    
    # TODO: these probably shouldn't be global variables..
    CALC_SIGMA = args.enable_sigmafit
    DATA_DIR = args.datadir
    
    # mount raid0 via sshfs on chiniak... (to get write access)
    #mount_raid0()

    # parse date string and convert to datetime object
    starttime = datetime.datetime(*time.strptime(args.starttime, "%Y.%m.%d.%H%M")[:7])
    endtime = datetime.datetime(*time.strptime(args.endtime, "%Y.%m.%d.%H%M")[:7])

    # sanity check arguements
    if args.recordlen > 24 or args.recordlen <= 0:
        print 'recordlen arguement must be greater than 0 hours and less than or equal to 24 hours'
        return
    if starttime > endtime:
        print 'start time is after end time..'
        return
    
    # compile list of start time/end time/radar/lock tuples 
    manager = Manager()
    lock = manager.Lock()
    records = []

    for radar in args.radars:
        print 'adding ' + radar + ' jobs to pool'
        if not radar in ['ksr.a', 'ade.a', 'adw.a', 'sps.a',  'kod.c', 'kod.d', 'mcm.a']:
            print radar + ' is not a UAF radar and may not have data on raid0, syncing with bigdipper...'
            cache_data(radar, starttime, endtime)

        stime = starttime
        while stime < endtime:
            etime = min(stime + datetime.timedelta(hours = args.recordlen), endtime)
            records.append((stime, etime, radar, lock))
            stime = etime
    
    # run pool of records in parallel
    # each worker takes 1 CPU and about 200 MB of GPU memory for 512x512 frequency/alpha resolution
    # so, two workers on kodiak-devel
    # i7 with a gtx970 could handle eight
    print 'starting fitlomb worker pool'
    fitlomb_pool = Pool(processes = args.poolsize)

    for record in records:
        if not DEBUG:
            fitlomb_pool.apply_async(func=generate_fitlomb, args=(record,))
        else:
            generate_fitlomb(record)

    fitlomb_pool.close()
    fitlomb_pool.join()
    print 'fitlomb workers finished...'

if __name__ == '__main__':
    main()


