/*
ECL-MaxFlow: This code computes the maximum flow of a directed or undirected graph.

Copyright (c) 2025, Avery VanAusdal and Martin Burtscher

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from
     this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at
https://github.com/burtscher/ECL-MaxFlow.

Publication: This work is described in detail in the following paper.
Avery Vanausdal and Martin Burtscher. An Efficient Push-Relabel Implementation for Max-Flow Computations on GPUs. Proceedings of the 44th IEEE International Performance Computing and Communications Conference. November 2025.

Sponsor: This code is based upon work supported by the U.S. National Science Foundation (NSF) under Award #1955367 and by an equipment donation from NVIDIA Corporation.
*/

#include <sys/time.h>
#include <vector>
#include <algorithm>
#include <limits.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include "ECLgraph.h"

typedef int excess_t;

static const int default_runs = 1;

static double GPUmaxflow(const ECLgraph g, const ECLgraph d_g, const int source, const int sink, int* const flow, int* const cap, const int* const rnindex, const int* const rnlist, const int* const retoe);

static double median(double array[], const int n)
{
  double median = 0;
  std::sort(array, array + n);
  if (n % 2 == 0) median = (array[(n - 1) / 2] + array[n / 2]) / 2.0;
  else median = array[n / 2];
  return median;
}

struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg);  cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg);  cudaEventDestroy(end);}
  void start() {cudaEventRecord(beg, 0);}
  double stop() {cudaEventRecord(end, 0);  cudaEventSynchronize(end);  float ms;  cudaEventElapsedTime(&ms, beg, end);  return 0.001 * ms;}
};

static int GPUinfo(const int d, bool print = false)
{
  cudaSetDevice(d);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, d);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {printf("ERROR: there is no CUDA capable device\n\n");  exit(-1);}
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int SMs = deviceProp.multiProcessorCount;
  printf("GPU: %s with %d SMs and %d mTpSM\n", deviceProp.name, SMs, mTpSM);
  return SMs * mTpSM;
}

static void CheckCuda()
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d: %s\n", e, cudaGetErrorString(e));
    exit(-1);
  }
}

template <typename T>
static __device__ inline T atomicRead(T* const addr)
{
  return ((cuda::atomic<T, cuda::thread_scope_device>*)addr)->load(cuda::memory_order_relaxed);
}

template <typename T>
static __device__ inline void atomicWrite(T* const addr, const T val)
{
  ((cuda::atomic<T, cuda::thread_scope_device>*)addr)->store(val, cuda::memory_order_relaxed);
}
