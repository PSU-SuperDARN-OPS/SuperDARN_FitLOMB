import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray 
import pycuda.compiler
import pycuda.autoinit
import numpy as np
import numexpr as ne
from itertools import chain, izip

# debugging imports
#import pdb
#import matplotlib.pyplot as plt

# todo: 
#       add second moment based error calculations

from timecube import make_spacecube
mod = pycuda.compiler.SourceModule("""
#include <stdio.h>
#include <stdint.h>

#define REAL 0
#define IMAG 1


// spawn one calc_bayes per frequency.., this makes it easier to parallelize dot products
// then, condense down to lag calculations for other stuff
// assuming...
// cs_matrix is [freq][alpha][time]

// RI - range * freq * alpha 
__global__ void calc_bayes(float *samples, float *lags, float *ce_matrix, float *se_matrix, float *cs_f, float *R_f, float *I_f, double *hbar2, double *P_f, float env_model, uint32_t nsamples, uint32_t nalphas, float dbar2)
{
    // assume cubecache already exists in memory..
    uint16_t t, i;
    uint32_t samplebase = blockIdx.x * nsamples * 2; 
    uint32_t RI_offset = (blockIdx.x * blockDim.x * nalphas) + (threadIdx.x * nalphas);

    for(i =  0; i < nalphas; i++) {
        /*if(threadIdx.x == 0) {
            printf("thread %d alpha %d\n", threadIdx.x, i);
        }*/
        uint32_t CS_offset = (threadIdx.x * nalphas * nsamples) + (i * nalphas);

        R_f[RI_offset + i] = 0;
        I_f[RI_offset + i] = 0;

        for(t = 0; t < nsamples; t++) {
            uint32_t sample_offset = samplebase + 2*t;
            if(lags[t]) { 
                // TODO: TRANSPOSE!!
                R_f[RI_offset + i] +=   samples[sample_offset + REAL] * ce_matrix[CS_offset + t] + \
                                        samples[sample_offset + IMAG] * ce_matrix[CS_offset + t];
                I_f[RI_offset + i] +=   samples[sample_offset + REAL] * se_matrix[CS_offset + t] - \
                                        samples[sample_offset + IMAG] * se_matrix[CS_offset + t];
            }
        }

        hbar2[RI_offset + i] = (pow(R_f[RI_offset + i],2) / cs_f[CS_offset + i] + \
                                pow(I_f[RI_offset + i],2) / cs_f[CS_offset + i]) / 2;

        P_f[RI_offset + i] = log10(nsamples * 2 * dbar2 - hbar2[RI_offset + i]) * (1 - nsamples) - log10(cs_f[CS_offset + i]);
    }
    __syncthreads();
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
   

if __name__ == '__main__':
    # prep synthetic data to process
    num_records=1
    fs = 100.
    ts = 1./fs

    f = [30.]
    amp = [10.]
    alf = [10.]
    env_model = 1
    lags = np.arange(0, 50) * ts
    times=[]
    signal=[]

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
   
    samples=np.float32(list(chain.from_iterable(izip(np.real(signal), np.imag(signal)))))
    times=np.array(np.float32(times))
    freqs = np.linspace(-fs/2, fs/2, 50)
    alfs = np.linspace(0,fs/2., 20)
    ce_matrix, se_matrix, CS_f = make_spacecube(times, freqs, alfs, env_model)
    ce_matrix = np.float32(ce_matrix)
    se_matrix = np.float32(se_matrix)
    CS_f = np.float32(CS_f)

    R_f = np.float32(np.zeros([len(freqs), len(alfs)]))
    I_f = np.float32(np.zeros([len(freqs), len(alfs)]))
    hbar2 = np.float64(np.zeros([len(freqs), len(alfs)]))
    P_f = np.float64(np.zeros([len(freqs), len(alfs)]))

    # allocate space on GPU 
    samples_gpu = cuda.mem_alloc(samples.nbytes)
    t_gpu = cuda.mem_alloc(times.nbytes)
    ce_gpu = cuda.mem_alloc(ce_matrix.nbytes)
    se_gpu = cuda.mem_alloc(se_matrix.nbytes)
    CS_f_gpu = cuda.mem_alloc(CS_f.nbytes)
    R_f_gpu = cuda.mem_alloc(CS_f.nbytes)
    I_f_gpu = cuda.mem_alloc(CS_f.nbytes)
    hbar2_gpu  = cuda.mem_alloc(2 * CS_f.nbytes)
    P_f_gpu = cuda.mem_alloc(2 * CS_f.nbytes)

    # copy over samples..
    cuda.memcpy_htod(samples_gpu, samples)
    cuda.memcpy_htod(t_gpu, times)

    cuda.memcpy_htod(ce_gpu, ce_matrix.flatten())
    cuda.memcpy_htod(se_gpu, se_matrix.flatten())
    cuda.memcpy_htod(CS_f_gpu, CS_f)

    # run kernel on GPU
    bayes_gpu = mod.get_function('calc_bayes')
    nsamples = np.int32(len(samples))
    nalphas = np.int32(len(alfs))
    dbar2 = np.float32((sum(np.real(samples) ** 2) + sum(np.imag(samples) ** 2)) / (nsamples))
    bayes_gpu(samples_gpu, t_gpu, ce_gpu, se_gpu, CS_f_gpu, R_f_gpu, I_f_gpu, hbar2_gpu, P_f_gpu, np.float32(env_model), nsamples, nalphas, dbar2, block = (1,1,1))

    # copy back data
    cuda.memcpy_dtoh(R_f, R_f_gpu) 
    cuda.memcpy_dtoh(I_f, I_f_gpu) 
    cuda.memcpy_dtoh(hbar2, hbar2_gpu)
    cuda.memcpy_dtoh(P_f, P_f_gpu)

    # run on CPU
    R_f_cpu, I_f_cpu, hbar2_cpu, P_f_cpu = calculate_bayes(signal, lags, freqs, alfs, env_model, ce_matrix, se_matrix, CS_f)

    # compare..
    import pdb
    pdb.set_trace()
