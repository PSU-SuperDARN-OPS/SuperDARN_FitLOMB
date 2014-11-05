#!/usr/bin/python2
import pycuda.driver as cuda
import pycuda.compiler
import pycuda.autoinit
import numpy as np
from itertools import chain, izip
from timecube import make_spacecube

# debugging imports
import pdb
import matplotlib.pyplot as plt
# todo: add second moment based error calculations

C = 3e8
FWHM_TO_SIGMA = 2.355 # conversion of fwhm to std deviation, assuming gaussian
LAMBDA_FIT = 1
SIGMA_FIT = 2

mod = pycuda.compiler.SourceModule("""
#include <stdio.h>
#include <stdint.h>

#define REAL 0
#define IMAG 1
#define MAX_SAMPLES 25
#define MAX_ALPHAS 128 // MUST BE A POWER OF 2
#define MAX_FREQS 128 // MUST BE A POWER OF 2
#define PI (3.141592)

// see generalizing the lomb-scargle periodogram, g. bretthorst
__global__ void calc_bayes(float *samples, int32_t *lags, float *alphas, float *lag_times, float *ce_matrix, float *se_matrix, double *P_f, float env_model, int32_t nsamples, int32_t nalphas, int32_t *n_good_lags_v)
{
    int32_t t, i, sample_offset, samplebase;
    double dbar2 = 0;
    double hbar2 = 0;
    int32_t n_good_samples = 0;
    float alpha;

    __shared__ float s_samples[MAX_SAMPLES * 2];
    __shared__ int32_t s_lags[MAX_SAMPLES];
    __shared__ float s_cs_f[MAX_ALPHAS];

     // parallel cache lag mask in shared memory
    samplebase = blockIdx.x * nsamples; 
    for(i = 0; i < nsamples / blockDim.x + 1; i++) {
        sample_offset = threadIdx.x + i * blockDim.x;
        if(sample_offset < nsamples) {
            s_lags[sample_offset] = (lags[samplebase + sample_offset] != 0);
        }
    }
    __syncthreads(); 

    // parallel cache samples in shared memory, each thread loading sample number tidx.x + n * nfreqs
    // mask out bad lags with zero
    samplebase = blockIdx.x * nsamples * 2; 
    for(i = 0; i < 2 * nsamples / blockDim.x + 1; i++) {
        sample_offset = threadIdx.x + i * blockDim.x;
        if(sample_offset < nsamples * 2) {
            s_samples[sample_offset] = samples[samplebase + sample_offset] * (s_lags[sample_offset >> 1] != 0);
        }
    }
    __syncthreads(); 
    
    // calculate number of *good* lags.. needed for dbar2 scaling
    for(i = 0; i < nsamples; i++) {
        if(s_lags[i]) {
            n_good_samples++;
        }
    }
    
    // parallel calculate cs_f given bad lags, cache in shared memory and store in global memory for later use (assumes nfreqs >= nalphas!!)
    // fastest if nfreqs == nalphas, avoids divering program counters between threads
    // TODO: add support for different environment models (e.g. sigma fit)
 
    if(threadIdx.x < nalphas) {
        s_cs_f[threadIdx.x] = 0;
        alpha = alphas[threadIdx.x];
        for(i = 0; i < nsamples; i++) {
            s_cs_f[threadIdx.x] += pow(exp(-alpha * lag_times[i]),2) * (s_lags[i] != 0);
        }
    }
    __syncthreads(); 

    // calculate dbar2 
    for(i = 0; i < 2*nsamples; i+=2) {
        dbar2 += (pow(s_samples[i + REAL],2) + pow(s_samples[i + IMAG],2)) * s_lags[i >> 1];
    }
    dbar2 /= 2 * n_good_samples;
    __syncthreads(); 
    

    // RI[pulse][alpha][freq]
    // CS[alpha][time][freq]
    for(i =  0; i < nalphas; i++) {
        int32_t RI_offset = (blockIdx.x * blockDim.x * nalphas) + (i * blockDim.x) + threadIdx.x;
        float r_f = 0;
        float i_f = 0;

        
        for(t = 0; t < nsamples; t++) {
            int32_t CS_offset = (i * blockDim.x * nsamples) + (t * blockDim.x) + threadIdx.x;
            sample_offset = 2*t;

            r_f += s_samples[sample_offset + REAL] * ce_matrix[CS_offset] + \
                   s_samples[sample_offset + IMAG] * se_matrix[CS_offset];
            i_f += s_samples[sample_offset + REAL] * se_matrix[CS_offset] - \
                   s_samples[sample_offset + IMAG] * ce_matrix[CS_offset];
        }

        hbar2 = ((pow(r_f, 2) / s_cs_f[i]) + (pow(i_f, 2) / s_cs_f[i]));
        P_f[RI_offset] = log10(n_good_samples * 2 * dbar2 - hbar2) * (1 - ((double) n_good_samples)) - log10(s_cs_f[i]);
    }

    if(threadIdx.x == 0) {
        n_good_lags_v[blockIdx.x] = n_good_samples;
    }
}

// P_f is [pulse][alpha][freq]
// thread for each freq, block across pulses
// TODO: currently assumes a power of 2 number of freqs 
__global__ void find_peaks(double *P_f, int32_t *peaks, int32_t nalphas)
{
    int32_t i;
    __shared__ int32_t maxidx[MAX_FREQS];
    __shared__ double maxval[MAX_FREQS];

    maxidx[threadIdx.x] = 0;
    maxval[threadIdx.x] = -1e6;
    
    // find max along frequency axis
    for(i = 0; i < nalphas; i++) {
        int32_t P_f_idx = (blockIdx.x * blockDim.x * nalphas) + (i * blockDim.x) + threadIdx.x;

        if (P_f[P_f_idx] > maxval[threadIdx.x]) { 
            maxidx[threadIdx.x] = P_f_idx;
            maxval[threadIdx.x] = P_f[P_f_idx];
        }
    }

    __syncthreads();
    // parallel reduce maximum
    for(i = blockDim.x/2; i > 0; i >>=1) {
        if(threadIdx.x < i) {
           if(maxval[threadIdx.x + i] > maxval[threadIdx.x]) {
              maxval[threadIdx.x] = maxval[threadIdx.x + i];
              maxidx[threadIdx.x] = maxidx[threadIdx.x + i];
           }
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) {
        peaks[blockIdx.x] = maxidx[threadIdx.x];
    }
}

// assumes nalphas == nfreqs..
// segment peak on P_f map

// find peak segment
// normalize peak segment probability
// multiply 
// threshold p_f on peak - factor
// for every point within glob above threshold
// calculate moment..?
//
// ---=------------
// ------===-------
// ------=+===-----
// ==------===-----
// ----------------
//

// start at peak idx
// traverse -> on each thread until below threshold
// for each thread, take total probability and prob * value
// traverse <- on each thread until below threshold
// for each thread, take total probability and prob * value
// parallel sum probability, prob * value
// increase prob * value by normalization of prob
//  

__global__ void moment_peaks(double *P_f, float *freq_peak, float *alf_peak, int32_t nalphas, float *freqs, float *alfs, float *alf_std, float *freq_std, double *amplitudes) 
{
    int32_t i;
    __shared__ float s_freqs[MAX_FREQS];
    __shared__ float s_alfs[MAX_ALPHAS];

    __shared__ float alf_value[MAX_ALPHAS];
    __shared__ float alf_prob[MAX_ALPHAS];
    __shared__ float freq_value[MAX_FREQS];
    __shared__ float freq_prob[MAX_FREQS];

    s_freqs[threadIdx.x] = freqs[threadIdx.x];
    s_alfs[threadIdx.x] = alfs[threadIdx.x];
    
    // compute frequency along max along frequency axis
    for(i = 0; i < nalphas; i++) {
        int32_t P_f_idx = (blockIdx.x * blockDim.x * nalphas) + (i * blockDim.x) + threadIdx.x;
        float p_f = P_f[P_f_idx];

        // TODO: figure out how to do this in log probabilities..
        alf_value[threadIdx.x] += s_alfs[i] * p_f;
        freq_value[threadIdx.x] += s_freqs[threadIdx.x] * p_f;
        alf_prob[threadIdx.x] += p_f;
        freq_prob[threadIdx.x] += p_f;
    }

    __syncthreads();
    // parallel reduce values and probabilities 
    /*
    for(i = blockDim.x/2; i > 0; i >>=1) {
        if(threadIdx.x < i) {
           if(maxval[threadIdx.x + i] > maxval[threadIdx.x]) {
              maxval[threadIdx.x] = maxval[threadIdx.x + i];
              maxidx[threadIdx.x] = maxidx[threadIdx.x + i];
           }
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) {
        freq_peak[blockIdx.x] = freq_value[0];
        freq_std[blockIdx.x] = freq_prob[0];
        alf_peak[blockIdx.x] = alf_value[0];
        alf_std[blockIdx.] = alf_prob[0];
    }

    */
}


// thread for each pulse, find fwhm and calculate ampltitude
__global__ void process_peaks(float *samples, float *ce_matrix, float *se_matrix, float *lag_times, float *freqs, float *alfs, double *P_f, float *snr, int32_t *lagmask, int32_t *n_good_lags, int32_t *peaks, int32_t nfreqs, int32_t nalphas, int32_t nlags, int32_t *alphafwhm, int32_t *freqfwhm, double *amplitudes) 
{
    int32_t peakidx = peaks[threadIdx.x];
    int32_t i;
    float fitpwr = 0;
    float rempwr = 0;
    float cs_f = 0;
    double apex = P_f[peakidx];

    float peakamp;
    float peakfreq;
    float peakalf;

    float fitted_signal[MAX_SAMPLES];
    float factor = (apex - .30103); // -.30103 is log10(.5)
     
    int32_t ffwhm = 1;
    int32_t afwhm = 1;
     
    int32_t pulse_lowerbound = peakidx - (peakidx % (nfreqs * nalphas));
    int32_t pulse_upperbound = pulse_lowerbound + (nfreqs * nalphas);
    __shared__ float s_times[MAX_SAMPLES]; 
    
    
    for(i = 0; i <= nlags / blockDim.x; i++) {
        int32_t idx = i * blockDim.x + threadIdx.x;
        if(idx < nlags) {
            s_times[idx] = lag_times[idx];
        }
    }
    __syncthreads();  

    // find alpha fwhm 
    for(i = peakidx; i < pulse_upperbound && P_f[i] > factor; i+=nfreqs) {
        afwhm++; 
    } 
    __syncthreads();  

    for(i = peakidx; i >= pulse_lowerbound && P_f[i] > factor; i-=nfreqs) {
        afwhm++; 
    }
    __syncthreads();  

    // find freq fwhm
    // don't care about fixing edge cases with peak on max or min freq, they are thrown as non-quality fits anyways
    for(i = peakidx; i % nfreqs != 0 && P_f[i] > factor; i++) {
        ffwhm++; 
    }
    __syncthreads();  

    for(i = peakidx; i % nfreqs != 0 && P_f[i] > factor; i--) {
        ffwhm++; 
    }
    __syncthreads();  // sync threads, they probably diverged during fwhm calculations

    int32_t alfidx = ((peakidx - (peakidx % nfreqs)) % (nfreqs * nalphas)) / nfreqs;
    int32_t freqidx = peakidx % nfreqs;
    peakfreq = freqs[freqidx];
    peakalf = alfs[alfidx];

    // calculate cs_f at peak, then calculate amplitude
    for(i = 0; i < nlags; i++) {
        cs_f += pow(exp(-peakalf * s_times[i]), 2) * (lagmask[threadIdx.x * nlags + i] != 0);
    }
    
    // recalculate r_f and i_f at peak (not caching eliminates many global memory accesses and reduces memory usage by 33%)
    float r_f = 0;
    float i_f = 0;
    
    for(i = 0; i < nlags; i++) {
        int32_t CS_offset = (alfidx * nfreqs * nlags) + (i * nfreqs) + freqidx;
        int32_t sample_offset = threadIdx.x * nlags * 2 + 2*i;

        r_f += samples[sample_offset + REAL] * ce_matrix[CS_offset] + \
               samples[sample_offset + IMAG] * se_matrix[CS_offset];
        i_f += samples[sample_offset + REAL] * se_matrix[CS_offset] - \
               samples[sample_offset + IMAG] * ce_matrix[CS_offset];
    }

    peakamp = (r_f + i_f) / cs_f;

    alphafwhm[threadIdx.x] = afwhm;
    freqfwhm[threadIdx.x] = ffwhm;
    amplitudes[threadIdx.x] = peakamp;
    
    // on each peak-thread, calculate fitted signal, fitted signal power, and remaining power
    for (i = 0; i < nlags; i++) {
        int32_t samplebase = threadIdx.x * nlags * 2; 

        float envelope = peakamp * exp(-peakalf * s_times[i]); 
        // TODO: add in environment model... exp(-peakalf ** 2?)
        float angle = 2 * PI * peakfreq * s_times[i];
        fitted_signal[2*i] = envelope * cos(angle);
        fitted_signal[2*i+1] = envelope * sin(angle);
        
        samples[samplebase + 2*i] -= fitted_signal[2*i];
        samples[samplebase + 2*i+1] -= fitted_signal[2*i+1];
        
        rempwr += sqrt(pow(samples[samplebase + 2*i],2) + pow(samples[samplebase + 2*i+1],2)) * lagmask[threadIdx.x * nlags + i];
        fitpwr += sqrt(pow(fitted_signal[2*i],2) + pow(fitted_signal[2*i+1],2)) * lagmask[threadIdx.x * nlags + i];
    }
    
    snr[threadIdx.x] = fitpwr / rempwr;
}



""")

# function to calculate P_f on CPU to check GPU calculations
def calculate_bayes(s, t, f, alfs, env_model):
    import numexpr as ne
    ce_matrix, se_matrix, CS_f = make_spacecube(t, f, alfs, env_model)

    N = len(t) * 2.# see equation (10) in [4]
    m = 2
    dbar2 = (sum(np.real(s) ** 2) + sum(np.imag(s) ** 2)) / (N) # (11) in [4] 
    R_f = (np.dot(np.real(s), ce_matrix) + np.dot(np.imag(s), se_matrix)).T
    I_f = (np.dot(np.real(s), se_matrix) - np.dot(np.imag(s), ce_matrix)).T
    
    hbar2 = ne.evaluate('((R_f ** 2) / CS_f + (I_f ** 2) / CS_f)')# (19) in [4] 
    
    P_f = np.log10(N * dbar2 - hbar2)  * ((2 - N) / 2.) - np.log10(CS_f)
    return R_f, I_f, hbar2, P_f

class BayesGPU:
    def __init__(self, lags, freqs, alfs, npulses, env_model):
        self.lags = np.float32(np.array(lags))
        self.freqs = np.float32(np.array(freqs))
        self.alfs = np.float32(np.array(alfs))
       
        self.npulses = npulses
        self.nlags = np.int32(len(self.lags))
        self.nalfs = np.int32(len(self.alfs))
        self.nfreqs = np.int32(len(self.freqs))
      
        self.env_model = np.float32(env_model)

        # do some sanity checks on the input parameters..
        if np.log2(self.nfreqs) != int(np.log2(self.nfreqs)):
            print 'ERROR: number of freqs should be a power of two'

        if self.nfreqs < self.nalfs:
            print 'ERROR: number of alfs should be less than number of freqs'
        
        if self.nfreqs > 1024:
            print 'ERROR: number of frequencies exceeds maximum thread size'
         
        if self.npulses > 1024:
            print 'ERROR: number of pulses exceeds maximum thread size'
         
        if self.npulses <= 1:
            print 'ERROR: number of pulses must be at least 2'

        # create matricies for processing
        ce_matrix, se_matrix, CS_f = make_spacecube(lags, freqs, alfs, env_model)
        ce_matrix_g = np.float32(np.swapaxes(ce_matrix,0,2)).flatten()
        se_matrix_g = np.float32(np.swapaxes(se_matrix,0,2)).flatten()
         
        # create dummy matricies to allocate on GPU
        lagmask = np.int32(np.zeros([self.npulses, self.nlags]))
        samples = np.float32(np.zeros([self.npulses, 2 * self.nlags]))

        self.P_f = np.float64(np.zeros([self.npulses, self.nalfs, self.nfreqs]))
        self.peaks = np.int32(np.zeros(self.npulses))
        self.alf_fwhm = np.int32(np.zeros(self.npulses))
        self.freq_fwhm = np.int32(np.zeros(self.npulses))
        self.amplitudes = np.float64(np.zeros(self.npulses))
        self.dbar2 = np.float64(np.zeros(self.npulses))
        self.snr = np.float32(np.zeros(self.npulses))
        self.n_good_lags = np.int32(np.zeros(self.npulses))
        
        # allocate space on GPU 
        self.samples_gpu = cuda.mem_alloc(samples.nbytes)
        self.lagmask_gpu = cuda.mem_alloc(lagmask.nbytes)
        self.ce_gpu = cuda.mem_alloc(ce_matrix_g.nbytes)
        self.se_gpu = cuda.mem_alloc(se_matrix_g.nbytes)
        self.P_f_gpu = cuda.mem_alloc(self.P_f.nbytes) # 450 mb
        self.peaks_gpu = cuda.mem_alloc(self.peaks.nbytes)
        self.alf_fwhm_gpu = cuda.mem_alloc(self.alf_fwhm.nbytes)
        self.freq_fwhm_gpu = cuda.mem_alloc(self.freq_fwhm.nbytes)
        self.amplitudes_gpu = cuda.mem_alloc(self.amplitudes.nbytes)
        self.n_good_lags_gpu = cuda.mem_alloc(self.n_good_lags.nbytes)
        self.snr_gpu = cuda.mem_alloc(self.snr.nbytes)
        self.lag_times_gpu = cuda.mem_alloc(self.lags.nbytes)
        self.freqs_gpu = cuda.mem_alloc(self.freqs.nbytes)
        self.alfs_gpu = cuda.mem_alloc(self.alfs.nbytes)
    
        # compute total GPU memory requirements..

        # copy ce/se/cs matricies over to GPU
        cuda.memcpy_htod(self.ce_gpu, ce_matrix_g)
        cuda.memcpy_htod(self.se_gpu, se_matrix_g)

        # copy over lags, frequencies, and alfs to GPU for SNR calculations
        cuda.memcpy_htod(self.lag_times_gpu, self.lags)
        cuda.memcpy_htod(self.freqs_gpu, self.freqs)
        cuda.memcpy_htod(self.alfs_gpu, self.alfs)

        # get cuda source modules
        self.calc_bayes = mod.get_function('calc_bayes')
        self.find_peaks = mod.get_function('find_peaks')
        self.process_peaks = mod.get_function('process_peaks')

    def run_bayesfit(self, samples, lagmask):
        self.lagmask = np.int32(lagmask)
        self.samples = samples
        cuda.memcpy_htod(self.samples_gpu, self.samples)
        cuda.memcpy_htod(self.lagmask_gpu, self.lagmask)
    
        # about 90% of the time is spent on calc_bayes
        self.calc_bayes(self.samples_gpu, self.lagmask_gpu, self.alfs_gpu, self.lag_times_gpu, self.ce_gpu, self.se_gpu, self.P_f_gpu, self.env_model, self.nlags, self.nalfs, self.n_good_lags_gpu, block = (int(self.nfreqs),1,1), grid = (int(self.npulses),1,1))
        self.find_peaks(self.P_f_gpu, self.peaks_gpu, self.nalfs, block = (int(self.nfreqs),1,1), grid = (int(self.npulses),1))
        self.process_peaks(self.samples_gpu, self.ce_gpu, self.se_gpu, self.lag_times_gpu, self.freqs_gpu, self.alfs_gpu, self.P_f_gpu, self.snr_gpu, self.lagmask_gpu, self.n_good_lags_gpu, self.peaks_gpu, self.nfreqs, self.nalfs, self.nlags, self.alf_fwhm_gpu, self.freq_fwhm_gpu, self.amplitudes_gpu, block = (int(self.npulses),1,1))

    
    def process_bayesfit(self, tfreq, noise):
        self.tfreq = tfreq
        self.noise = noise

        cuda.memcpy_dtoh(self.amplitudes, self.amplitudes_gpu)
        cuda.memcpy_dtoh(self.alf_fwhm, self.alf_fwhm_gpu)
        cuda.memcpy_dtoh(self.freq_fwhm, self.freq_fwhm_gpu)
        cuda.memcpy_dtoh(self.peaks, self.peaks_gpu)
        cuda.memcpy_dtoh(self.n_good_lags, self.n_good_lags_gpu)
        cuda.memcpy_dtoh(self.snr, self.snr_gpu)

        dalpha = self.alfs[1] - self.alfs[0]
        dfreqs = self.freqs[1] - self.freqs[0]
        
        N = 2 * self.n_good_lags
        
        w_idx = ((self.peaks - (self.peaks % self.nfreqs)) % (self.nfreqs * self.nalfs)) / self.nfreqs
        v_idx = self.peaks % self.nfreqs

        self.w = (self.alfs[w_idx] * C) / (2. * np.pi * (tfreq * 1e3))
        self.w_std = dalpha * (((C * self.alf_fwhm) / (2. * np.pi * (tfreq * 1e3))) / FWHM_TO_SIGMA)
        self.w_e = self.w_std / np.sqrt(N)
        
        self.v = (self.freqs[v_idx] * C) / (2 * tfreq * 1e3)
        self.v_std = dfreqs * ((((self.freq_fwhm) * C) / (2 * tfreq * 1e3)) / FWHM_TO_SIGMA)
        self.v_e = self.v_std / np.sqrt(N)

        self.p = self.amplitudes / noise
        self.p[self.p <= 0] = np.nan
        self.p = 10 * np.log10(self.p)
         
        # raw freq/decay for debugging
        self.vfreq = (self.freqs[v_idx])
        self.walf = (self.alfs[w_idx])

    # pickle a pulse for later analysis
    def pickle_pulse(self, filename='pulse.p'):
        import pickle
        param = {}
        param['alfs'] = self.alfs
        param['freqs'] = self.freqs
        param['samples'] = self.samples
        param['lagmask'] = self.lagmask
        param['lags'] = self.lags
        param['npulses'] = self.npulses
        param['env_model'] = self.env_model
        param['tfreq'] = self.tfreq
        param['noise'] = self.noise
        pickle.dump(param, open(filename, 'wb'))


# example function to run cuda_bayes against some generated data
# to profile, add @profile atop interesting functions       
# run kernprof -l cuda_bayes.py
# then python -m line_profiler cuda_bayes.py.lprof to view results
def main():
    SYNTHETIC = False 

    f = [2.9]
    amp = [27000.]
    alf = [33.5]

    if SYNTHETIC:
        fs = 100.
        ts = 1./fs

        lags = np.arange(0, 24) * ts
        signal=[]

        maxpulses = 2
        tfreq = 30 
        noise = 1e-6#0.001
        env_model = LAMBDA_FIT 
        lmask = [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0]
        lagmask = np.tile(lmask, maxpulses)
        lagmask = np.tile(np.ones(len(lags)), maxpulses)
        
        F = f[0]+noise*np.random.randn()
        U = alf[0]+noise*np.random.randn()
        A = amp[0]+noise*np.random.rand()
       

        for t,lag in enumerate(lags):
            N_I=noise*np.random.randn()
            N_Q=noise*np.random.randn()
            N_A=np.sqrt(N_I**2+N_Q**2)
            N_phase=np.tan(N_Q/N_I)
            N_phase=0#np.tan(N_Q/N_I)
            sig=A * np.exp(1j * 2 * np.pi * F * lag) * np.exp(-U * lag)#N_A*np.exp(1j*N_phase)
            signal.append(sig)
        
        samples = np.tile(np.float32(list(chain.from_iterable(izip(np.real(signal), np.imag(signal))))), maxpulses)

        freqs = np.linspace(-fs/2, fs/2, 16)
        alfs = np.linspace(0,fs/2, 16)

    else: 
        import pickle
        param = pickle.load(open('pulse.p', 'rb'))
        lags = param['lags']
        freqs = param['freqs']
        alfs = param['alfs']
        alfs = alfs[::4]
        freqs = freqs[::4]
        maxpulses = param['npulses']
        env_model = param['env_model']
        samples = param['samples']
        lagmask = param['lagmask']
        tfreq = param['tfreq']
        noise = param['noise']

    gpu = BayesGPU(lags, freqs, alfs, maxpulses, env_model)
    gpu.run_bayesfit(samples, lagmask)
    gpu.process_bayesfit(tfreq, noise)
    
    rgate = 11
    if SYNTHETIC: 
        rgate = 0
    print 'calculated amplitude: ' + str(gpu.amplitudes[rgate]) 
    print 'calculated freq: ' + str(gpu.vfreq[rgate]) 
    print 'calculated decay: ' + str(gpu.walf[rgate]) 
    
    print 'snr: ' + str(gpu.snr[rgate])
    if not SYNTHETIC: 
        i_samp = samples[rgate][0::2]
        q_samp = samples[rgate][1::2]
        #plt.plot(gpu.lags[np.nonzero(i_samp)], i_samp[np.nonzero(i_samp)])
        #plt.plot(gpu.lags[np.nonzero(q_samp)], q_samp[np.nonzero(q_samp)])
    else:
        pass
        #plt.plot(gpu.lags, np.real(signal))
        #plt.plot(gpu.lags, np.imag(signal))

    print 'actual decay: ' + str(alf[0])
    print 'actual freq: ' + str(f[0])
    print 'actual amplitude: ' + str(amp[0])

    #samples_cpu = np.array(samples[0:int(len(samples)/maxpulses):2]) + 1j * np.array(samples[1:int(len(samples)/maxpulses):2])
    #R_f_cpu, I_f_cpu, P_f_cpu = calculate_bayes(samples_cpu, lags, freqs, alfs, env_model)
    
    fit = gpu.amplitudes[rgate] * np.exp(1j * 2 * np.pi * gpu.vfreq[rgate] * lags) * np.exp(-gpu.walf[rgate] * lags)
    expected = amp[0] * np.exp(1j * 2 * np.pi * f[0] * lags) * np.exp(-alf[0] * lags)
    #plt.plot(gpu.lags, np.real(fit))
    #plt.plot(gpu.lags, np.imag(fit))

    #plt.plot(lags, np.real(expected))
    #plt.plot(lags, np.imag(expected))
    
    #plt.plot(lags, amp[0] * ce_matrix[0][:,7] + .05)
    #plt.legend(['samp_i', 'samp_q', 'fit_i', 'fit_q', 'expected_i', 'expected_q', 'ce_matrix'])
    #plt.show()
    


    ''' 
    if max(P_f_cpu.flatten() - gpu.P_f.flatten()[0:(len(freqs) * len(alfs))]) < 3e-6:
        print 'P_f on GPU and CPU match'
    else:
        print 'P_f calculation error! GPU and CPU matricies do not match'
    '''

    cuda.memcpy_dtoh(gpu.P_f, gpu.P_f_gpu)
    #pdb.set_trace()
    for gate in range(75):
        print gate
        print 'calculated amplitude: ' + str(gpu.amplitudes[gate]) 
        print 'calculated snr: ' + str(gpu.snr[gate]) 
        print 'max P_f: ' + str(max(gpu.P_f[gate].flatten()))
        print 'min P_f: ' + str(min(gpu.P_f[gate].flatten()))
        # so, scale by P_f[0], renormalize again later
        p_f = gpu.P_f[gate]
        p_f -= p_f[0][0] # normalize off first element
        p_f = 10 ** p_f
        p_f /= sum(p_f.flatten())

        fprob = np.sum(p_f, axis=0)
        aprob = np.sum(p_f, axis=1)

        print 'calculated freq: ' + str(gpu.vfreq[gate]) 
        print 'mom freq: ' + str(sum(freqs * fprob))

        print 'calculated decay: ' + str(gpu.walf[gate]) 
        print 'mom alf: ' + str(sum(alfs * aprob))

     
        plt.imshow(gpu.P_f[gate])
        plt.show()
    # peak about 18 260
    # freq 260
    # alf 18

def moment_peak(P_f, alphas, freqs):
    return 0, 0, 0, 0
if __name__ == '__main__':
    main()



