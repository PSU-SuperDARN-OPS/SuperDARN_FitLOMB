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

__global__ void calc_bayes(float *samples, float *lags, float *ce_matrix, float *se_matrix, float *cs_f, float *R_f, float *I_f, double *hbar2, double *P_f, float env_model, uint32_t nsamples, uint32_t nalphas)
{
    // possible optimizations:
    // - eliminate cs_f freq dependence..
    uint32_t t, i, sample_offset;
    double dbar2 = 0;
    


    // cache samples in shared memory
    __shared__ float s_samples[100];
    __shared__ float s_cs_f[100];
    
    uint32_t samplebase = blockIdx.x * nsamples * 2; 

    for(i = 0; i < 2 * nsamples / blockDim.x + 1; i++) {
        sample_offset = threadIdx.x + i * blockDim.x;

        if(sample_offset < nsamples * 2) {
            s_samples[sample_offset] = samples[samplebase + sample_offset];
        
        }
    }

    // assumes nfreqs > nalphas.. 
    if(threadIdx.x < nalphas) {
        s_cs_f[threadIdx.x] = cs_f[threadIdx.x];
    }
    
    __syncthreads(); 
    // calculate dbar on thread zero.. (or do on CPU?, takes about 1% of time))
    if(threadIdx.x == 0) {
        for(i = 0; i < 2*nsamples; i+=2) {
            dbar2 += pow(s_samples[i + REAL],2) + pow(s_samples[i + IMAG],2);
        }
        dbar2 /= 2 * nsamples;
    }
    __syncthreads();  // probably not *needed*..


    // RI[pulse][alpha][freq]
    // CS[alpha][time][freq]
    // 90% of time spent on this loop..
    for(i =  0; i < nalphas; i++) {
        uint32_t RI_offset = (blockIdx.x * blockDim.x * nalphas) + (i * blockDim.x) + threadIdx.x;
        R_f[RI_offset] = 0;
        I_f[RI_offset] = 0;

        // TODO: if T is a good lag..
        // ~80% of time here
        for(t = 0; t < nsamples; t++) {
            uint32_t CS_offset = (i * blockDim.x * nsamples) + (t * blockDim.x) + threadIdx.x;
            sample_offset = 2*t;

            R_f[RI_offset] +=   s_samples[sample_offset + REAL] * ce_matrix[CS_offset] + \
                                s_samples[sample_offset + IMAG] * se_matrix[CS_offset];
            I_f[RI_offset] +=   s_samples[sample_offset + REAL] * se_matrix[CS_offset] - \
                                s_samples[sample_offset + IMAG] * ce_matrix[CS_offset];
        }

        // ~20% of time
        hbar2[RI_offset] = (pow(R_f[RI_offset],2) / s_cs_f[i] + \
                            pow(I_f[RI_offset],2) / s_cs_f[i]) / 2;

        P_f[RI_offset] = log10(nsamples * 2 * dbar2 - hbar2[RI_offset]) * (1 - ((int32_t) nsamples)) - log10(s_cs_f[i]);
    }
}
""")

def calculate_bayes(s, t, f, alfs, env_model, ce_matrix, se_matrix, CS_f):
    N = len(t) * 2.# see equation (10) in [4]
    m = 2

    dbar2 = (sum(np.real(s) ** 2) + sum(np.imag(s) ** 2)) / (N) # (11) in [4] 
    print 'dbar2: ' + str(dbar2)
    R_f = (np.dot(np.real(s), ce_matrix) + np.dot(np.imag(s), se_matrix)).T
    I_f = (np.dot(np.real(s), se_matrix) - np.dot(np.imag(s), ce_matrix)).T
    
    hbar2 = ne.evaluate('((R_f ** 2) / CS_f + (I_f ** 2) / CS_f) / 2.')# (19) in [4] 
    
    P_f = np.log10(N * dbar2 - hbar2)  * ((2 - N) / 2.) - np.log10(CS_f)
    pdb.set_trace()
    return R_f, I_f, hbar2, P_f
   
#@profile
# kernprof -l cuda_bayes.py
#python -m line_profiler script_to_profile.py.lprof
def main():
    # prep synthetic data to process
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
    CUDA_GRID = 1

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
    freqs = np.linspace(-fs/2, fs/2, 100)
    freqs = np.linspace(0, fs/2, 100)
    alfs = np.linspace(0,fs/2., 60)

    times=np.array(np.float32(times))
    ce_matrix_cpu, se_matrix_cpu, CS_f_cpu = make_spacecube(times, freqs, alfs, env_model)
    times=np.tile(np.array(np.float32(times)), CUDA_GRID)
    ce_matrix, se_matrix, CS_f = make_spacecube(times, freqs, alfs, env_model)
    # TODO: RESHAPE CE/SE MATRIX FOR FREQ/ALPHA/TIME (swap 2,3?)
    # 50 50 20 (
    ce_matrix_g = np.float32(np.swapaxes(ce_matrix,0,2)).flatten()
    se_matrix_g = np.float32(np.swapaxes(se_matrix,0,2)).flatten()
    CS_f_g = np.float32(np.swapaxes(CS_f,0,1).flatten())
    CS_f = np.float32(CS_f)

    R_f = np.float32(np.zeros([CUDA_GRID, len(alfs), len(freqs)]))
    I_f = np.float32(np.zeros([CUDA_GRID, len(alfs), len(freqs)]))
    hbar2 = np.float64(np.zeros([CUDA_GRID, len(alfs), len(freqs)]))
    P_f = np.float64(np.zeros([CUDA_GRID, len(alfs), len(freqs)]))

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

    # copy over samples..
    cuda.memcpy_htod(samples_gpu, samples)
    cuda.memcpy_htod(t_gpu, times)
    cuda.memcpy_htod(ce_gpu, ce_matrix_g)
    cuda.memcpy_htod(se_gpu, se_matrix_g)
    cuda.memcpy_htod(CS_f_gpu, CS_f_g)
    gpu_start = time.time()
    bayes_gpu = mod.get_function('calc_bayes')
    nsamples = np.int32(len(samples) / 2 / CUDA_GRID)
    nalphas = np.int32(len(alfs))

    for i in range(1):
        # run kernel on GPU
        bayes_gpu(samples_gpu, t_gpu, ce_gpu, se_gpu, CS_f_gpu, R_f_gpu, I_f_gpu, hbar2_gpu, P_f_gpu, np.float32(env_model), nsamples, nalphas,  block = (int(len(freqs)),1,1), grid = (CUDA_GRID,1))
    for i in range(1):
        # copy back data
        cuda.memcpy_dtoh(R_f, R_f_gpu) 
        cuda.memcpy_dtoh(I_f, I_f_gpu) 
        cuda.memcpy_dtoh(hbar2, hbar2_gpu)
        cuda.memcpy_dtoh(P_f, P_f_gpu)
    
    R_f = R_f[0]
    I_f = I_f[0] 

    gpu_end = time.time()

    cpu_start = time.time()
    for i in range(1):
        R_f_cpu, I_f_cpu, hbar2_cpu, P_f_cpu = calculate_bayes(signal, lags, freqs, alfs, env_model, ce_matrix_cpu, se_matrix_cpu, CS_f_cpu)

    cpu_end = time.time()
    print 'gpu: ' + str(gpu_end - gpu_start)
    print 'cpu: ' + str(cpu_end - cpu_start)
    # compare..
    pdb.set_trace()

if __name__ == '__main__':
    main()


