#!/usr/bin/python2
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray 
import pycuda.compiler
import pycuda.autoinit
import numpy as np
import numexpr as ne
from itertools import chain, izip
import time
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
#define MAX_FREQS 256 // MUST BE A POWER OF 2

// lags is mask of good lags 
__global__ void calc_bayes(float *samples, int32_t *lags, float *ce_matrix, float *se_matrix, float *cs_f, float *R_f, float *I_f, double *hbar2, double *P_f, float env_model, int32_t nsamples, int32_t nalphas, double *dbar2_v, int32_t *n_good_lags_v)
{
    int32_t t, i, sample_offset, samplebase;
    double dbar2 = 0;
    int32_t n_good_samples = 0;

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
    
    // parallel cache cs_f to shared memory (assumes nfreqs > nalphas!!!)
    if(threadIdx.x < nalphas) {
        s_cs_f[threadIdx.x] = cs_f[threadIdx.x];
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
        R_f[RI_offset] = 0;
        I_f[RI_offset] = 0;
        for(t = 0; t < nsamples; t++) {
            int32_t CS_offset = (i * blockDim.x * nsamples) + (t * blockDim.x) + threadIdx.x;
            sample_offset = 2*t;

            R_f[RI_offset] +=   s_samples[sample_offset + REAL] * ce_matrix[CS_offset] + \
                                s_samples[sample_offset + IMAG] * se_matrix[CS_offset];
            I_f[RI_offset] +=   s_samples[sample_offset + REAL] * se_matrix[CS_offset] - \
                                s_samples[sample_offset + IMAG] * ce_matrix[CS_offset];
        }

        hbar2[RI_offset] = (pow(R_f[RI_offset],2) / s_cs_f[i] + \
                            pow(I_f[RI_offset],2) / s_cs_f[i]) / 2;
        
        P_f[RI_offset] = log10(nsamples * 2 * dbar2 - hbar2[RI_offset]) * (1 - ((int32_t) nsamples)) - log10(s_cs_f[i]);
    }

    if(threadIdx.x == 0) {
        dbar2_v[blockIdx.x] = dbar2;
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

// thread for each pulse, find fwhm and calculate ampltitude
__global__ void process_peaks(double *P_f, float *R_f, float *I_f, float *CS_f, double *dbar2, float *snr, int32_t *lagmask, int32_t *n_good_lags, float *cs_lagpwr, int32_t *peaks, int32_t nfreqs, int32_t nalphas, int32_t nlags, int32_t *alphafwhm, int32_t *freqfwhm, double *amplitudes) 
{
    int32_t peakidx = peaks[threadIdx.x];
    int32_t i;
    float fitpwr = 0;

    double apex = P_f[peakidx];
    
    float factor = (apex - .30103); // -.30103 is log10(.5)
     
    int32_t ffwhm = 1;
    int32_t afwhm = 1;
     
    int32_t pulse_lowerbound = peakidx - (peakidx % (nfreqs * nalphas));
    int32_t pulse_upperbound = pulse_lowerbound + (nfreqs * nalphas);
    
    // find alpha fwhm, change formatting to more direct..
    for(i = peakidx; i < pulse_upperbound && P_f[i] > factor; i+=nfreqs) {
        afwhm++; 
    } 
    for(i = peakidx; i >= pulse_lowerbound && P_f[i] > factor; i-=nfreqs) {
        afwhm++; 
    }
    // find freq fwhm
    // don't care about fixing edge cases with peak on max or min freq, they are thrown out anyways

    for(i = peakidx; i % nfreqs != 0 && P_f[i] > factor; i++) {
        ffwhm++; 
    }
     
    for(i = peakidx; i % nfreqs != 0 && P_f[i] > factor; i--) {
        ffwhm++; 
    }
    __syncthreads();  // sync threads, they probably diverged during fwhm calculations
    
    alphafwhm[threadIdx.x] = afwhm;
    freqfwhm[threadIdx.x] = ffwhm;
    amplitudes[threadIdx.x] = (R_f[peakidx] + I_f[peakidx]) / CS_f[peakidx % nalphas];

    int32_t cs_lagpwr_idx = (peakidx % (nfreqs * nalphas)) * nlags;
    
    // calculate mean power in fitted signal
    // incorrect.. should be ratio of fitted signal and samples without fitted signal
    for(i = 0; i < nlags; i++) {
        fitpwr += lagmask[threadIdx.x * nlags + i] * cs_lagpwr[cs_lagpwr_idx + i];
    }
    fitpwr /= n_good_lags[threadIdx.x];
    fitpwr *= pow(((float) amplitudes[threadIdx.x]) / 2, 2); // scale cached cs_lagpwr for calculated amplitude (TODO: TEST..)
    snr[threadIdx.x] = fitpwr / (2 * dbar2[threadIdx.x]);
}
""")

def calculate_bayes(s, t, f, alfs, env_model, ce_matrix, se_matrix, CS_f):
    N = len(t) * 2.# see equation (10) in [4]
    m = 2

    dbar2 = (sum(np.real(s) ** 2) + sum(np.imag(s) ** 2)) / (N) # (11) in [4] 
    R_f = (np.dot(np.real(s), ce_matrix) + np.dot(np.imag(s), se_matrix)).T
    I_f = (np.dot(np.real(s), se_matrix) - np.dot(np.imag(s), ce_matrix)).T
    
    hbar2 = ne.evaluate('((R_f ** 2) / CS_f + (I_f ** 2) / CS_f) / 2.')# (19) in [4] 
    
    P_f = np.log10(N * dbar2 - hbar2)  * ((2 - N) / 2.) - np.log10(CS_f)
    return R_f, I_f, hbar2, P_f

class BayesGPU:
    def __init__(self, lags, freqs, alfs, npulses, env_model):
        self.lags = np.array(lags)
        self.freqs = np.array(freqs)
        self.alfs = np.array(alfs)
       
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
        CS_f_g = np.float32(np.swapaxes(CS_f,0,1).flatten())
         
        cs_lagpwr = np.float32((ce_matrix.swapaxes(1,2).swapaxes(0,1) ** 2) + (se_matrix.swapaxes(1,2).swapaxes(0,1) ** 2)).flatten() # freq alf lag
        # create dummy matricies to allocate on GPU
        lagmask = np.int32(np.zeros([self.npulses, self.nlags]))
        R_f = np.float32(np.zeros([self.npulses, self.nalfs, self.nfreqs]))
        I_f = np.float32(np.zeros([self.npulses, self.nalfs, self.nfreqs]))
        hbar2 = np.float64(np.zeros([self.npulses, self.nalfs, self.nfreqs]))
        P_f = np.float64(np.zeros([self.npulses, self.nalfs, self.nfreqs]))
        samples = np.float32(np.zeros([self.npulses, 2 * self.nlags]))

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
        self.cs_lagpwr_gpu = cuda.mem_alloc(cs_lagpwr.nbytes)
        self.CS_f_gpu = cuda.mem_alloc(CS_f.nbytes)
        self.R_f_gpu = cuda.mem_alloc(R_f.nbytes)
        self.I_f_gpu = cuda.mem_alloc(I_f.nbytes)
        self.hbar2_gpu  = cuda.mem_alloc(hbar2.nbytes)
        self.P_f_gpu = cuda.mem_alloc(P_f.nbytes)
        self.peaks_gpu = cuda.mem_alloc(self.peaks.nbytes)
        self.alf_fwhm_gpu = cuda.mem_alloc(self.alf_fwhm.nbytes)
        self.freq_fwhm_gpu = cuda.mem_alloc(self.freq_fwhm.nbytes)
        self.amplitudes_gpu = cuda.mem_alloc(self.amplitudes.nbytes)
        self.dbar2_gpu = cuda.mem_alloc(self.dbar2.nbytes)
        self.n_good_lags_gpu = cuda.mem_alloc(self.n_good_lags.nbytes)
        self.snr_gpu = cuda.mem_alloc(self.snr.nbytes)

        # copy ce/se/cs matricies over to GPU
        cuda.memcpy_htod(self.ce_gpu, ce_matrix_g)
        cuda.memcpy_htod(self.se_gpu, se_matrix_g)
        cuda.memcpy_htod(self.CS_f_gpu, CS_f_g)
        cuda.memcpy_htod(self.cs_lagpwr_gpu, cs_lagpwr)
    
        # get cuda source modules
        self.calc_bayes = mod.get_function('calc_bayes')
        self.find_peaks = mod.get_function('find_peaks')
        self.process_peaks = mod.get_function('process_peaks')

    def run_bayesfit(self, samples, lagmask):
        lagmask = np.int32(lagmask)
        cuda.memcpy_htod(self.samples_gpu, samples)
        cuda.memcpy_htod(self.lagmask_gpu, lagmask)

        self.calc_bayes(self.samples_gpu, self.lagmask_gpu, self.ce_gpu, self.se_gpu, self.CS_f_gpu, self.R_f_gpu, self.I_f_gpu, self.hbar2_gpu, self.P_f_gpu, self.env_model, self.nlags, self.nalfs, self.dbar2_gpu, self.n_good_lags_gpu, block = (int(self.nfreqs),1,1), grid = (int(self.npulses),1,1))
        self.find_peaks(self.P_f_gpu, self.peaks_gpu, self.nalfs, block = (int(self.nfreqs),1,1), grid = (int(self.npulses),1))
        self.process_peaks(self.P_f_gpu, self.R_f_gpu, self.I_f_gpu, self.CS_f_gpu, self.dbar2_gpu, self.snr_gpu, self.lagmask_gpu, self.n_good_lags_gpu, self.cs_lagpwr_gpu, self.peaks_gpu, self.nfreqs, self.nalfs, self.nlags, self.alf_fwhm_gpu, self.freq_fwhm_gpu, self.amplitudes_gpu, block = (int(self.npulses),1,1))

        cuda.memcpy_dtoh(self.amplitudes, self.amplitudes_gpu)
        cuda.memcpy_dtoh(self.alf_fwhm, self.alf_fwhm_gpu)
        cuda.memcpy_dtoh(self.freq_fwhm, self.freq_fwhm_gpu)
        cuda.memcpy_dtoh(self.peaks, self.peaks_gpu)
        cuda.memcpy_dtoh(self.dbar2, self.dbar2_gpu)
        cuda.memcpy_dtoh(self.n_good_lags, self.n_good_lags_gpu)
        cuda.memcpy_dtoh(self.snr, self.snr_gpu)

    def process_bayesfit(self, tfreq, noise):
        dalpha = self.alfs[1] - self.alfs[0]
        dfreqs = self.freqs[1] - self.freqs[0]
        
        # TODO: calculate SNR
        N = 2 * self.n_good_lags
        
        w_l_idx = ((self.peaks - (self.peaks % self.nfreqs)) % (self.nfreqs * self.nalfs)) / self.nfreqs
        v_l_idx = self.peaks % self.nfreqs

        self.w_l = (self.alfs[w_l_idx] * C) / (2. * np.pi * (tfreq * 1e3))
        self.w_l_std = dalpha * (((C * self.alf_fwhm) / (2. * np.pi * (tfreq * 1e3))) / FWHM_TO_SIGMA)
        self.w_l_e = self.w_l_std / np.sqrt(N)
        
        self.v_l = (self.freqs[v_l_idx] * C) / (2 * tfreq * 1e3)
        self.v_l_std = dfreqs * ((((self.freq_fwhm) * C) / (2 * tfreq * 1e3)) / FWHM_TO_SIGMA)
        self.v_l_e = self.v_l_std / np.sqrt(N)

        self.p_l = self.amplitudes / noise
        self.p_l[self.p_l <= 0] = np.nan
        self.p_l = 10 * np.log10(self.p_l)

        
#@profile
# kernprof -l cuda_bayes.py
#python -m line_profiler script_to_profile.py.lprof
def main():
    fs = 100.
    ts = 1./fs

    f = [30.]
    amp = [20.]
    alf = [10.]
    env_model = 1
    lags = np.arange(0, 25) * ts
    times=[]
    signal=[]

    MAXPULSES = 100
    TFREQ = 30 
    NOISE = 10
    
    lagmask = np.tile(np.ones(len(lags)), MAXPULSES)
    
    F = f[0]+0.1*np.random.randn()
    U = alf[0]+0.1*np.random.randn()
    A = amp[0]+0.1*np.random.rand()
    for t,lag in enumerate(lags):
        times.append(lag)
        N_I=NOISE*np.random.randn()
        N_Q=NOISE*np.random.randn()
        N_A=np.sqrt(N_I**2+N_Q**2)
        N_phase=np.tan(N_Q/N_I)
        sig=A * np.exp(1j * 2 * np.pi * F * lag) * np.exp(-U * lag)+N_A*np.exp(1j*N_phase)
        signal.append(sig)

    samples = np.tile(np.float32(list(chain.from_iterable(izip(np.real(signal), np.imag(signal))))), MAXPULSES)

    freqs = np.linspace(-fs/2, fs/2, 256)
    alfs = np.linspace(0,fs/2., 128)
    
    gpu = BayesGPU(lags, freqs, alfs, MAXPULSES, LAMBDA_FIT)
    gpu.run_bayesfit(samples, lagmask)
    gpu.process_bayesfit(TFREQ, NOISE)
   
    pdb.set_trace()

if __name__ == '__main__':
    main()



