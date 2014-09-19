#!/usr/bin/python2
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray 
import pycuda.compiler
import pycuda.autoinit
import numpy as np
import numexpr as ne
from itertools import chain, izip

# debugging imports
import pdb
import matplotlib.pyplot as plt
import time
# todo: 
#       add second moment based error calculations

from timecube import make_spacecube
mod = pycuda.compiler.SourceModule("""

#include <stdio.h>
#include <stdint.h>

#define REAL 0
#define IMAG 1
#define MAX_NRANG 300
#define MAX_SAMPLES 30 
#define MAX_ALPHAS 64

// TODO: FIX FOR GRID > 1...
__global__ void calc_bayes(float *samples, float *lags, float *ce_matrix, float *se_matrix, float *cs_f, float *R_f, float *I_f, double *hbar2, double *P_f, float env_model, int32_t nsamples, int32_t nalphas)
{
    int32_t t, i, sample_offset;
    double dbar2 = 0;

    __shared__ float s_samples[MAX_SAMPLES * 2];
    __shared__ float s_cs_f[MAX_ALPHAS];
    
    // parallel cache samples in shared memory, each thread loading sample number tidx.x + n * nfreqs
    int32_t samplebase = blockIdx.x * nsamples * 2; 
    for(i = 0; i < 2 * nsamples / blockDim.x + 1; i++) {
        sample_offset = threadIdx.x + i * blockDim.x;

        if(sample_offset < nsamples * 2) {
            s_samples[sample_offset] = samples[samplebase + sample_offset];
        
        }
    }

    // parallel cache cs_f to shared memory (assumes nfreqs > nalphas!!!)
    if(threadIdx.x < nalphas) {
        s_cs_f[threadIdx.x] = cs_f[threadIdx.x];
    }
    __syncthreads(); 
    
    // calculate dbar2 
    for(i = 0; i < 2*nsamples; i+=2) {
        dbar2 += pow(s_samples[i + REAL],2) + pow(s_samples[i + IMAG],2);
    }
    dbar2 /= 2 * nsamples;
    __syncthreads(); 
    

    // RI[pulse][alpha][freq]
    // CS[alpha][time][freq]
    for(i =  0; i < nalphas; i++) {
        int32_t RI_offset = (blockIdx.x * blockDim.x * nalphas) + (i * blockDim.x) + threadIdx.x;
        R_f[RI_offset] = 0;
        I_f[RI_offset] = 0;
        // TODO: if T is a good lag..
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
}

// P_f is [pulse][alpha][freq]
// thread for each freq, block across pulses
// TODO: currently assumes a power of 2 number of freqs 
__global__ void find_peaks(double *P_f, int32_t *peaks, int32_t nalphas)
{
    int32_t i;
    __shared__ int32_t maxidx[MAX_NRANG];
    __shared__ double maxval[MAX_NRANG];

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
__global__ void process_peaks(double *P_f, float *R_f, float *I_f, float *CS_f, int32_t *peaks, int32_t nfreqs, int32_t nalphas, int32_t *alphafwhm, int32_t *freqfwhm, double *amplitudes) 
{
    int32_t peakidx = peaks[threadIdx.x];
    int32_t i;

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

    alphafwhm[threadIdx.x] = afwhm;
    freqfwhm[threadIdx.x] = ffwhm;

    amplitudes[threadIdx.x] = (R_f[peakidx] + I_f[peakidx]) / CS_f[peakidx % nalphas];
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
   
#@profile
# kernprof -l cuda_bayes.py
#python -m line_profiler script_to_profile.py.lprof
def main():
    num_records=1
    fs = 100.
    ts = 1./fs

    f = [30.]
    amp = [10.]
    alf = [10.]
    env_model = 1
    lags = np.arange(0, 25) * ts
    times=[]
    signal=[]
    CUDA_GRID = 75

    for i in xrange(num_records):
      F=f[0]+0.1*np.random.randn()+float(i-num_records/2)/float(num_records)*.1*f[0]
      U=alf[0]+0.1*np.random.randn()
      A=amp[0]+0.1*np.random.rand()
      for t,lag in enumerate(lags):
        times.append(lag)
        N_I=0.1*np.random.randn()
        N_Q=0.1*np.random.randn()
        N_A=np.sqrt(N_I**2+N_Q**2)
        N_phase=np.tan(N_Q/N_I)
        sig=A * np.exp(1j * 2 * np.pi * F * lag) * np.exp(-U * lag)+N_A*np.exp(1j*N_phase)
        signal.append(sig)
   
    samples=np.tile(np.float32(list(chain.from_iterable(izip(np.real(signal), np.imag(signal))))), CUDA_GRID)
    freqs = np.linspace(-fs/2, fs/2, 128)
    #freqs = np.linspace(0, fs/2, 128)
    alfs = np.linspace(0,fs/2., 64)

    times=np.array(np.float32(times))
    ce_matrix_cpu, se_matrix_cpu, CS_f_cpu = make_spacecube(times, freqs, alfs, env_model)
    ce_matrix, se_matrix, CS_f = make_spacecube(times, freqs, alfs, env_model)
  
    times=np.tile(np.array(np.float32(times)), CUDA_GRID)
    ce_matrix_g = np.float32(np.swapaxes(ce_matrix,0,2)).flatten()
    se_matrix_g = np.float32(np.swapaxes(se_matrix,0,2)).flatten()
    CS_f_g = np.float32(np.swapaxes(CS_f,0,1).flatten())
    CS_f = np.float32(CS_f)

    R_f = np.float32(np.zeros([CUDA_GRID, len(alfs), len(freqs)]))
    I_f = np.float32(np.zeros([CUDA_GRID, len(alfs), len(freqs)]))
    hbar2 = np.float64(np.zeros([CUDA_GRID, len(alfs), len(freqs)]))
    P_f = np.float64(np.zeros([CUDA_GRID, len(alfs), len(freqs)]))
        
    peaks = np.int32(np.zeros(CUDA_GRID))
    alf_fwhm = np.int32(np.zeros(CUDA_GRID))
    freq_fwhm = np.int32(np.zeros(CUDA_GRID))
    amplitudes = np.float64(np.zeros(CUDA_GRID))

    # allocate space on GPU 
    samples_gpu = cuda.mem_alloc(samples.nbytes)
    t_gpu = cuda.mem_alloc(times.nbytes)
    ce_gpu = cuda.mem_alloc(ce_matrix_g.nbytes)
    se_gpu = cuda.mem_alloc(se_matrix_g.nbytes)
    CS_f_gpu = cuda.mem_alloc(CS_f.nbytes)
    R_f_gpu = cuda.mem_alloc(R_f.nbytes)
    I_f_gpu = cuda.mem_alloc(I_f.nbytes)
    hbar2_gpu  = cuda.mem_alloc(hbar2.nbytes)
    P_f_gpu = cuda.mem_alloc(P_f.nbytes)
    peaks_gpu = cuda.mem_alloc(peaks.nbytes)
    alf_fwhm_gpu = cuda.mem_alloc(alf_fwhm.nbytes)
    freq_fwhm_gpu = cuda.mem_alloc(freq_fwhm.nbytes)
    amplitudes_gpu = cuda.mem_alloc(amplitudes.nbytes)

    cuda.memcpy_htod(R_f_gpu, R_f)
    cuda.memcpy_htod(I_f_gpu, I_f)
    cuda.memcpy_htod(P_f_gpu, P_f)
    cuda.memcpy_htod(hbar2_gpu, hbar2)

    cuda.memcpy_htod(t_gpu, times)
    cuda.memcpy_htod(ce_gpu, ce_matrix_g)
    cuda.memcpy_htod(se_gpu, se_matrix_g)
    cuda.memcpy_htod(CS_f_gpu, CS_f_g)

    calc_bayes = mod.get_function('calc_bayes')
    find_peaks = mod.get_function('find_peaks')
    process_peaks = mod.get_function('process_peaks')

    gpu_start = time.time()
    nsamples = np.int32(len(samples) / 2 / CUDA_GRID)
    nalphas = np.int32(len(alfs))
    nfreqs = np.int32(len(freqs))


    for i in range(500):
        # run kernel on GPU
        cuda.memcpy_htod(samples_gpu, samples)
        calc_bayes(samples_gpu, t_gpu, ce_gpu, se_gpu, CS_f_gpu, R_f_gpu, I_f_gpu, hbar2_gpu, P_f_gpu, np.float32(env_model), nsamples, nalphas,  block = (int(nfreqs),1,1), grid = (CUDA_GRID,1,1))
        find_peaks(P_f_gpu, peaks_gpu, nalphas, block = (int(nfreqs),1,1), grid = (CUDA_GRID,1))
        #process_peaks(P_f_gpu, R_f_gpu, I_f_gpu, CS_f_gpu, peaks_gpu, nfreqs, nalphas, alf_fwhm_gpu, freq_fwhm_gpu, amplitudes_gpu, block = (CUDA_GRID,1,1))
 
        # copy back data
        cuda.memcpy_dtoh(amplitudes, amplitudes_gpu)
        cuda.memcpy_dtoh(alf_fwhm, alf_fwhm_gpu)
        cuda.memcpy_dtoh(freq_fwhm, freq_fwhm_gpu)
        cuda.memcpy_dtoh(peaks, peaks_gpu)
    
    gpu_end = time.time()

    cpu_start = time.time()
    for i in range(1):
        R_f_cpu, I_f_cpu, hbar2_cpu, P_f_cpu = calculate_bayes(signal, lags, freqs, alfs, env_model, ce_matrix_cpu, se_matrix_cpu, CS_f_cpu)

    cpu_end = time.time()
    print 'gpu: ' + str(gpu_end - gpu_start)
    print 'cpu: ' + str(cpu_end - cpu_start)

    # compare..
    '''
    plt.subplot(411)
    plt.imshow(P_f[0], interpolation="nearest")
    plt.subplot(412)
    plt.imshow(P_f[1], interpolation="nearest") 
    plt.subplot(413)
    plt.imshow(P_f_cpu, interpolation="nearest")

    plt.subplot(414)
    plt.imshow(P_f[0] - P_f_cpu, interpolation="nearest")

    plt.show()
    pdb.set_trace()
    '''

if __name__ == '__main__':
    main()


