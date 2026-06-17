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

#include <cuda_runtime.h>
#include <cuda/atomic>
#include "maxflow.h"

static const int ThreadsPerBlock = 512;
static const int WarpSize = 32;
static const int maxval = INT_MAX;
static excess_t g_last_maxflow_result = -1;

static __global__ __launch_bounds__(ThreadsPerBlock)
void init(const ECLgraph g, const int source, const int sink, int* const flow, excess_t* const excess, int* const time, int* const height)
{
  const int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  
  if (idx < g.edges) {
    flow[idx] = 0;
  }
  
  if (idx < g.nodes) {
    excess[idx] = 0;
    time[idx] = 0;
    height[idx] = (idx == sink) ? 0 : g.nodes;  // sink = 0, others = g.nodes, like Derigs & Meier (1989)
  }
}

// sync: excess, wlsize
// safe: flow
static __global__ __launch_bounds__(ThreadsPerBlock)
void initialPush(const ECLgraph g, const int source, const int sink, excess_t* const excess, int* const flow, const int* const cap, 
                                    int* const wl, int* const wlsize)
{
  const int e = g.nindex[source] + threadIdx.x + blockIdx.x * ThreadsPerBlock;
  
  if (e < g.nindex[source + 1]) {
    const int capacity = cap[e];
    const int dst = g.nlist[e];
    flow[e] = capacity;
    atomicAdd(&excess[source], -capacity);  // subtract from source
    atomicAdd(&excess[dst], capacity);
    if (dst != sink) {
      wl[atomicAdd(wlsize, 1)] = dst;
    }
  }
}

static __global__ __launch_bounds__(ThreadsPerBlock)
void reverseResidualBFS(const ECLgraph g, const int* const flow, const int* const cap, 
                                    const int* const rnindex, const int* const rnlist, const int* const retoe, 
                                    const int* const wl1, const int wl1size, int* const wl2, int* const wl2size, 
                                    int* const time, const int iter, 
                                    const int source, const int sink, int* const height, const int curr_h)
{
  const int idx = (threadIdx.x + blockIdx.x * ThreadsPerBlock) / WarpSize;  // full warp per worklist item
  const int lane = threadIdx.x % WarpSize;
  
  if (idx < wl1size) {
    const int v = wl1[idx];
    
    // follow incoming edges backwards
    const int rbeg = rnindex[v];
    const int rend = rnindex[v + 1];
    for (int re = rbeg + lane; re < rend; re += WarpSize) {  // distribute edges among threads in the warp
      const int e = retoe[re];
      const int n = rnlist[re];
      
      if (n != source && n != sink && flow[e] < cap[e]) {  // only operate on residual edges (have spare capacity) 
        if (atomicMin(&time[n], iter) != iter) {  // if n hasn't been visited yet in this BFS
          height[n] = curr_h;
          wl2[atomicAdd(wl2size, 1)] = n;
        }
      }
    }
    
    // follow outgoing edges if there's undoable flow (backwards wrt the mirrored residual edge) 
    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    for (int e = beg + lane; e < end; e += WarpSize) {  // distribute edges among threads in the warp
      const int n = g.nlist[e];
      
      if (n != source && n != sink && flow[e] > 0) {  // only operate on residual edges (have flow available to undo) 
        if (atomicMin(&time[n], iter) != iter) {  // if n hasn't been visited yet in this BFS
          height[n] = curr_h;
          wl2[atomicAdd(wl2size, 1)] = n;
        }
      }
    }
  }
}

static __global__ __launch_bounds__(ThreadsPerBlock)
void phase2_reverseResidualBFS(const ECLgraph g, const int* const flow, const int* const cap, 
                                    const int* const rnindex, const int* const rnlist, const int* const retoe, 
                                    const int* const wl1, const int wl1size, int* const wl2, int* const wl2size, 
                                    int* const time, const int iter, 
                                    const int source, const int sink, int* const height, const int curr_h)
{
  const int idx = (threadIdx.x + blockIdx.x * ThreadsPerBlock) / WarpSize;  // full warp per worklist item
  const int lane = threadIdx.x % WarpSize;
  
  if (idx < wl1size) {
    const int v = wl1[idx];
    
    // follow outgoing edges if there's undoable flow (backwards wrt the mirrored residual edge) 
    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    for (int e = beg + lane; e < end; e += WarpSize) {  // distribute edges among threads in the warp
      const int n = g.nlist[e];
      
      if (n != source && n != sink && flow[e] > 0) {  // only operate on residual edges (have flow available to undo) 
        if (atomicMin(&time[n], iter) != iter) {  // if n hasn't been visited yet in this BFS
          height[n] = curr_h;
          wl2[atomicAdd(wl2size, 1)] = n;
        }
      }
    }
  }
}

// phase 1: only nodes with excess > 0 and height < |V| are active, only get flow to the sink
// sync: height, flow, excess, time, wl2size
static __global__ void phase1_pushRelabel(const ECLgraph g, const int source, const int sink, 
                        int* const height, excess_t* const excess, 
                        int* const flow, const int* const cap, 
                        const int* const rnindex, const int* const rnlist, const int* const retoe, 
                        int* const wl1, const int wl1size, int* const wl2, int* const wl2size,
                        int* const time, const int iter)
{
  const int idx = (threadIdx.x + blockIdx.x * ThreadsPerBlock) / WarpSize;  // full warp per worklist item
  const int lane = threadIdx.x % WarpSize;
  
  if (idx < wl1size) {
    const int v = wl1[idx];
    const int v_height = atomicRead(&height[v]);
    
    if (v_height < g.nodes && atomicRead(&excess[v]) > 0) {  // if vertex v is active in Phase 1
      int min_height = INT_MAX;
      int min_dst;
      int min_e;
      bool minIsForward = true;
      
      // follow outgoing edges (to push excess flow away)
      const int beg = g.nindex[v];
      const int end = g.nindex[v + 1];
      for (int e = beg + lane; e < end; e += WarpSize) {  // distribute edges among threads in the warp
        const int dst = g.nlist[e];
        if (atomicRead(&flow[e]) < cap[e]) {  // residual capacity > 0 (has room to add outgoing flow)
          const int dst_height = atomicRead(&height[dst]);
          if (dst_height < min_height) {
            min_height = dst_height;
            min_dst = dst;
            min_e = e;
          }
        }
      }  // fwd edges checked
      
      // follow incoming edges backwards if there's undoable incoming flow (forward wrt the mirrored residual edge)
      // to undo flow away from v
      const int rbeg = rnindex[v];
      const int rend = rnindex[v + 1];
      for (int re = rbeg + lane; re < rend; re += WarpSize) {  // distribute edges among threads in the warp
        const int dst = rnlist[re];
        const int e = retoe[re];  // e points at v
        if (atomicRead(&flow[e]) > 0) {  // residual capacity > 0 (has removeable incoming flow)
          const int dst_height = atomicRead(&height[dst]);
          if (dst_height < min_height) {
            min_height = dst_height;
            min_dst = dst;
            min_e = e;
            minIsForward = false;
          }
        }
      }  // reverse edges checked
      
      // parallel reduction in the warp https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
      for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
        int new_h = __shfl_down_sync(~0, min_height, offset);
        int new_dst = __shfl_down_sync(~0, min_dst, offset);
        int new_e = __shfl_down_sync(~0, min_e, offset);
        bool new_isF = __shfl_down_sync(~0, minIsForward, offset);
        if (new_h < min_height) {
          min_height = new_h;
          min_dst = new_dst;
          min_e = new_e;
          minIsForward = new_isF;
        }
      }
      
      // only lane 0 does the pushing/relabeling
      if (lane == 0) {
        if (min_height < v_height) {  // if admissable (shortest neighbor is shorter than v)
          const int dst = min_dst;
          const int e = min_e;
          excess_t delta;
          if (minIsForward) {
            delta = min(atomicRead(&excess[v]), cap[e] - atomicRead(&flow[e]));  // add as much outgoing flow as we can to reach 0 excess
            atomicAdd(&flow[e], delta);  // adds flow towards dst
          } else {  // min_e is reverse edge
            delta = min(atomicRead(&excess[v]), atomicRead(&flow[e]));  // undo as much incoming flow as we can to reach 0 excess
            atomicAdd(&flow[e], -delta);  // flow drains back to dst
          }
          const int old_excess = atomicAdd(&excess[v], -delta);  // remove excess from v
          if (old_excess > delta && atomicMax(&time[v], iter) != iter) {  // if v still has remaining excess after subtracting delta...
            wl2[atomicAdd(wl2size, 1)] = v;  // re-add v to wl if it wasn't able to push away all excess
          }
          atomicAdd(&excess[dst], delta);
          if (dst != source && dst != sink && atomicMax(&time[dst], iter) != iter) {
            wl2[atomicAdd(wl2size, 1)] = dst;  // add the potentially re-activated dst node to wl
          }
        } else {
          if (min_height != maxval) {  // lift/relabel if no admissable out-arcs (still active)
            atomicWrite(&height[v], min_height + 1);  // set to minimum height for an admissable out-arc
            if (((min_height + 1) < g.nodes) && atomicMax(&time[v], iter) != iter) {
              wl2[atomicAdd(wl2size, 1)] = v;  // re-add v to wl, wasn't able to push away any excess
            }
          } else {  // found no residual neighbors at all
            atomicWrite(&height[v], g.nodes);
          }
        }
      }
    }
  }
}

// phase 2: any node with excess is active
// sync: height, flow, excess, time, wl2size
static __global__ void phase2_pushRelabel(const ECLgraph g, const int source, const int sink, 
                        int* const height, excess_t* const excess, 
                        int* const flow, const int* const cap, 
                        const int* const rnindex, const int* const rnlist, const int* const retoe, 
                        int* const wl1, const int wl1size, int* const wl2, int* const wl2size,
                        int* const time, const int iter)
{
  const int idx = (threadIdx.x + blockIdx.x * ThreadsPerBlock) / WarpSize;  // full warp per worklist item
  const int lane = threadIdx.x % WarpSize;
  
  if (idx < wl1size) {
    const int v = wl1[idx];
    
    if (atomicRead(&excess[v]) > 0) {  // if vertex v is active
      int min_height = INT_MAX;
      int min_dst;
      int min_e;
      
      // follow incoming edges backwards if there's undoable incoming flow (forward wrt the mirrored residual edge)
      // to undo flow away from v
      const int rbeg = rnindex[v];
      const int rend = rnindex[v + 1];
      for (int re = rbeg + lane; re < rend; re += WarpSize) {  // distribute edges among threads in the warp
        const int dst = rnlist[re];
        const int e = retoe[re];  // e points at v
        if (atomicRead(&flow[e]) > 0) {  // residual capacity > 0 (has removeable incoming flow)
          const int dst_height = atomicRead(&height[dst]);
          if (dst_height < min_height) {
            min_height = dst_height;
            min_dst = dst;
            min_e = e;
          }
        }
      }  // reverse edges checked
      
      // parallel reduction in the warp https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
      for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
        int new_h = __shfl_down_sync(~0, min_height, offset);
        int new_dst = __shfl_down_sync(~0, min_dst, offset);
        int new_e = __shfl_down_sync(~0, min_e, offset);
        if (new_h < min_height) {
          min_height = new_h;
          min_dst = new_dst;
          min_e = new_e;
        }
      }
      
      // only lane 0 does the pushing/relabeling
      if (lane == 0) {
        if (min_height < atomicRead(&height[v])) {  // if admissable (shortest neighbor is shorter than v)
          const int dst = min_dst;
          const int e = min_e;
          excess_t delta, residual;
          residual = atomicRead(&flow[e]);
          delta = min(atomicRead(&excess[v]), residual);  // undo as much incoming flow as we can to reach 0 excess
          atomicAdd(&flow[e], -delta);  // flow drains back to dst
          const int old_excess = atomicAdd(&excess[v], -delta);  // remove excess from v
          if (old_excess > delta && atomicMax(&time[v], iter) != iter) {  // if v still has remaining excess after subtracting delta...
            wl2[atomicAdd(wl2size, 1)] = v;  // re-add v to wl if it wasn't able to push away all excess
          }
          atomicAdd(&excess[dst], delta);
          if (dst != source && dst != sink && atomicMax(&time[dst], iter) != iter) {
            wl2[atomicAdd(wl2size, 1)] = dst;  // add the potentially re-activated dst node to wl
          }
        } else {
          if (min_height != maxval) {  // lift/relabel if no admissable out-arcs (still active)
            atomicWrite(&height[v], min_height + 1);  // set to minimum height for an admissable out-arc
          }
          if (atomicMax(&time[v], iter) != iter) {
              wl2[atomicAdd(wl2size, 1)] = v;  // re-add v to wl, wasn't able to push away any excess
          }
        }
      }
    }
  }
}

// phase 1: initialize BFS from sink; Lines 2-4 of Algorithm 6
static __global__ __launch_bounds__(ThreadsPerBlock)
void phase1_initGR(const ECLgraph g, const int source, const int sink, int* const height, int* const wl)
{
  const int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (idx < g.nodes) {
    height[idx] = (idx == sink) ? 0 : g.nodes;  // sink = 0, others = g.nodes, like Derigs & Meier (1989)
    if (idx == 0) wl[0] = sink;
  }
}

// phase 2: initialize BFS from source; Lines 6-8 of Algorithm 6
static __global__ __launch_bounds__(ThreadsPerBlock)
void phase2_initGR(const ECLgraph g, const int source, const int sink, int* const height, int* const wl)
{
  const int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (idx < g.nodes) {
    height[idx] = (idx == source) ? 0 : INT_MAX;  // source = 0, others = INT_MAX;
    if (idx == 0) wl[0] = source;
  }
}

// add active nodes for Phase 2 to WL_PR (excess > 0, non-sink, non-source)
static __global__ __launch_bounds__(ThreadsPerBlock)
void phase2_buildWL(const ECLgraph g, const int source, const int sink, int* const excess, int* const wl, int* const wlsize)
{
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < g.nodes && v != source && v != sink) {
    if (excess[v] > 0) {
      wl[atomicAdd(wlsize, 1)] = v;
    }
  }
}

// covers Lines 6-25 of Algorithm 4
static void launchPushRelabel(const ECLgraph d_g, const int source, const int sink, 
                              int* const d_height, excess_t* const d_excess, 
                              int* const d_flow, const int* const d_cap, 
                              const int* const d_rnindex, const int* const d_rnlist, const int* const d_retoe,
                              int* d_wl1, int wl1size, int* d_wl2, int* const d_wl2size, int* d_wl3, 
                              int* const d_time, const int gr_iters, int* const d_count, const int GR_frequency)
{
  GPUTimer timer, phaseTimer;
  double pushrelabelTime = 0.0;
  double bfsTime = 0.0;
  
  int blocks;
  const int node_blocks = (d_g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock;  // to launch one thread per vertex
  int wl3size;
  
  double sum_bfs_percent = 0.0;
  int bfs_iter_count = gr_iters;
  int GR_count = 0;
  
  phaseTimer.start();
  // phase 1 push-relabel loop: push flow to sink
  int iter = 0;
  do {
    iter++;
    timer.start();
    
    if (cudaSuccess != cudaMemset(d_wl2size, 0, sizeof(int))) {fprintf(stderr, "ERROR: setting d_wl2size to 0 failed\n"); exit(-1);}  // empties WL_next
    blocks = ((long)wl1size * WarpSize + ThreadsPerBlock - 1) / ThreadsPerBlock;  // full warp per worklist item
    
    // push-relabel step
    phase1_pushRelabel<<<blocks, ThreadsPerBlock>>>(d_g, source, sink, d_height, d_excess, d_flow, d_cap, d_rnindex, d_rnlist, d_retoe, d_wl1, wl1size, d_wl2, d_wl2size, d_time, iter);
    
    if (cudaSuccess != cudaMemcpy(&wl1size, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying of wl2size from device failed\n"); exit(-1);}
    std::swap(d_wl1, d_wl2);  // swap WL_PR and WL_next; Line 21 of Algorithm 4
    cudaDeviceSynchronize();
    pushrelabelTime += timer.stop();
    
    if (wl1size == 0) break;  // skip the global relabel if there are no more active vertices (i.e., Phase 1 is complete)
    
    // global relabel step
    if ((iter % GR_frequency) == 0) {
      GR_count++;
      timer.start();
      
      // reset heights for BFS, add sink to worklist WL_GR
      phase1_initGR<<<node_blocks, ThreadsPerBlock>>>(d_g, source, sink, d_height, d_wl3);
      wl3size = 1;
      
      // global relabel BFS loop
      int bfs_iter = 0; 
      do {
        bfs_iter++;
        
        cudaMemset(d_wl2size, 0, sizeof(int));  // re-use PR's secondary worklist WL_next
        blocks = ((long)wl3size * WarpSize + ThreadsPerBlock - 1) / ThreadsPerBlock;  // full warp per worklist item
        
        sum_bfs_percent += ((100.0 * wl3size) / d_g.nodes);
        bfs_iter_count++;
        
        // passes a negative "iter" value for d_time in BFS to avoid interfering with the Push-Relabel kernel's use of d_time; see Section 4 of paper
        reverseResidualBFS<<<blocks, ThreadsPerBlock>>>(d_g, d_flow, d_cap, d_rnindex, d_rnlist, d_retoe, d_wl3, wl3size, d_wl2, d_wl2size, d_time, -iter, source, sink, d_height, bfs_iter);
        
        if (cudaSuccess != cudaMemcpy(&wl3size, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of d_wl2size from device failed\n");
        std::swap(d_wl3, d_wl2);  // swap WL_GR and WL_next; Line 16 of Algorithm 6
      } while (wl3size > 0);
      
      const double time_taken = timer.stop();
      printf("Global relabel from sink on iter %i: %i iters in %.6fs\n", iter, bfs_iter, time_taken);
      bfsTime += time_taken;
    }
  } while (wl1size > 0);
  
  printf("---Phase 1 complete, minimum cut found in %.6fs ---\n", phaseTimer.stop());
  printf("\titerations: %i\n", iter);
  printf("\tpush-relabel kernel time: %.6fs\n", pushrelabelTime);
  printf("\tglobal relabeled %i times in %.6fs\n", GR_count, bfsTime);
  
  printf("\tavg percent of graph in BFS WL: %.2f%% over %i total BFS iters\n", sum_bfs_percent / (bfs_iter_count - gr_iters), (bfs_iter_count - gr_iters));  // subtract gr_iters because initial BFS doesn't track sum_bfs_percent, only bfs_iter_count
  
  // reset stat trackers for Phase 2
  const int phase1_pr_iters = iter;  // need to keep incrementing iter and bfs_iter_count for d_time to keep working, so save these
  const int phase1_bfs_iters = bfs_iter_count;
  pushrelabelTime = 0.0;
  bfsTime = 0.0;
  GR_count = 0;
  sum_bfs_percent = 0.0;
  
  phaseTimer.start();
  // get active nodes (excess > 0) for PR worklist
  phase2_buildWL<<<node_blocks, ThreadsPerBlock>>>(d_g, source, sink, d_excess, d_wl1, d_wl2size);
  if (cudaSuccess != cudaMemcpy(&wl1size, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of d_wl2size from device failed\n");
  
  if (wl1size == 0) {
    printf("No active nodes remaining, Phase 2 not needed!\n");
    return;
  }
  
  printf("---Phase 2 starts, returning excess flow from %i active nodes to source---\n", wl1size);
  // phase 2: pull remaining excess flow back to the source
  // run BFS from source to set initial heights for Phase 2
  GR_count++;
  timer.start();
  // reset heights for BFS, add source to worklist
  phase2_initGR<<<node_blocks, ThreadsPerBlock>>>(d_g, source, sink, d_height, d_wl3);
  wl3size = 1;
  
  int bfs_iter1 = 0; 
  do {
    bfs_iter1++;
    
    cudaMemset(d_wl2size, 0, sizeof(int));  // re-use PR's secondary worklist WL_next
    blocks = ((long)wl3size * WarpSize + ThreadsPerBlock - 1) / ThreadsPerBlock;  // full warp per worklist item
    
    sum_bfs_percent += ((100.0 * wl3size) / d_g.nodes);
    bfs_iter_count++;
    
    // passes a negative "iter" value for d_time in BFS to avoid interfering with the Push-Relabel kernel's use of d_time; see Section 4 of paper
    phase2_reverseResidualBFS<<<blocks, ThreadsPerBlock>>>(d_g, d_flow, d_cap, d_rnindex, d_rnlist, d_retoe, d_wl3, wl3size, d_wl2, d_wl2size, d_time, -iter, source, sink, d_height, bfs_iter1);
    
    if (cudaSuccess != cudaMemcpy(&wl3size, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of d_wl2size from device failed\n");
    std::swap(d_wl3, d_wl2);  // swap WL_GR and WL_next; Line 16 of Algorithm 6
  } while (wl3size > 0);
  
  const double time_taken = timer.stop();
  printf("initial global relabel from source on iter %i: %i iters in %.6fs\n", iter, bfs_iter1, time_taken);
  bfsTime += time_taken;
  
  // phase 2 push-relabel loop: pull remaining flow back to source
  while (wl1size > 0) {
    iter++;  // continue incrementing global PR iter count for d_time to work without reset
    timer.start();
    
    if (cudaSuccess != cudaMemset(d_wl2size, 0, sizeof(int))) {fprintf(stderr, "ERROR: setting d_wl2size to 0 failed\n"); exit(-1);}  // empties WL_next
    blocks = ((long)wl1size * WarpSize + ThreadsPerBlock - 1) / ThreadsPerBlock;  // full warp per worklist item
    
    // push-relabel step
    phase2_pushRelabel<<<blocks, ThreadsPerBlock>>>(d_g, source, sink, d_height, d_excess, d_flow, d_cap, d_rnindex, d_rnlist, d_retoe, d_wl1, wl1size, d_wl2, d_wl2size, d_time, iter);
    
    if (cudaSuccess != cudaMemcpy(&wl1size, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying of wl2size from device failed\n"); exit(-1);}
    std::swap(d_wl1, d_wl2);  // swap WL_PR and WL_next; Line 21 of Algorithm 4
    cudaDeviceSynchronize();
    pushrelabelTime += timer.stop();
    
    if (wl1size == 0) break;  // skip the global relabel if there are no more active vertices (i.e., Phase 2 is complete)
    
    // global relabel step
    if (((iter - phase1_pr_iters) % GR_frequency) == 0) {
      GR_count++;
      timer.start();
      
      // reset heights for BFS, add source to worklist WL_GR
      phase2_initGR<<<node_blocks, ThreadsPerBlock>>>(d_g, source, sink, d_height, d_wl3);
      wl3size = 1;
      
      // global relabel BFS loop
      int bfs_iter = 0; 
      do {
        bfs_iter++;
        
        cudaMemset(d_wl2size, 0, sizeof(int));  // re-use PR's secondary worklist WL_next
        blocks = ((long)wl3size * WarpSize + ThreadsPerBlock - 1) / ThreadsPerBlock;  // full warp per worklist item
        
        sum_bfs_percent += ((100.0 * wl3size) / d_g.nodes);
        bfs_iter_count++;
        
        // passes a negative "iter" value for d_time in BFS to avoid interfering with the Push-Relabel kernel's use of d_time; see Section 4 of paper
        phase2_reverseResidualBFS<<<blocks, ThreadsPerBlock>>>(d_g, d_flow, d_cap, d_rnindex, d_rnlist, d_retoe, d_wl3, wl3size, d_wl2, d_wl2size, d_time, -iter, source, sink, d_height, bfs_iter);
        
        if (cudaSuccess != cudaMemcpy(&wl3size, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of d_wl2size from device failed\n");
        std::swap(d_wl3, d_wl2);  // swap WL_GR and WL_next; Line 16 of Algorithm 6
      } while (wl3size > 0);
      
      const double time_taken2 = timer.stop();
      printf("Global relabel from source on iter %i: %i iters in %.6fs\n", iter, bfs_iter, time_taken2);
      bfsTime += time_taken2;
    }
  }
  printf("---Phase 2 complete, full flow state found in %.6fs ---\n", phaseTimer.stop());
  printf("\titerations: %i\n", iter - phase1_pr_iters);
  printf("\tpush-relabel kernel time: %.6fs\n", pushrelabelTime);
  printf("\tglobal relabeled %i times in %.6fs\n", GR_count, bfsTime);
  const int phase2_bfs_iters = bfs_iter_count - phase1_bfs_iters;
  printf("\tavg percent of graph in BFS WL: %.2f%% over %i total BFS iters\n", sum_bfs_percent / phase2_bfs_iters, phase2_bfs_iters);
  
}

// launches BFS from sink to set initial height values; Line 4 of Algorithm 4
static int launchInitialBFS(const ECLgraph d_g, const int source, const int sink, 
                          int* const d_height, int* const d_flow, const int* const d_cap, 
                          const int* const d_rnindex, const int* const d_rnlist, const int* const d_retoe, 
                          int* d_wl3, int* d_wl4, int* const d_wl4size, 
                          int* const d_time)
{
  GPUTimer timer;
  timer.start();
  
  int blocks;
  
  // start BFS from sink for Phase 1 initial heights
  if (cudaSuccess != cudaMemcpy(d_wl3, &sink, sizeof(int), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: adding sink to d_wl3 failed\n");
  int wl3size = 1;
  
  int iter = 0;
  do {
    iter++;
    
    cudaMemset(d_wl4size, 0, sizeof(int)); // empties WL_next
    blocks = ((long)wl3size * WarpSize + ThreadsPerBlock - 1) / ThreadsPerBlock;  // full warp per worklist item
    
    // uses BFS_ID == -1
    reverseResidualBFS<<<blocks, ThreadsPerBlock>>>(d_g, d_flow, d_cap, d_rnindex, d_rnlist, d_retoe, 
                                                            d_wl3, wl3size, d_wl4, d_wl4size, d_time, -1, source, sink, d_height, iter);
    
    if (cudaSuccess != cudaMemcpy(&wl3size, d_wl4size, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of wl4size from device failed\n");
    std::swap(d_wl3, d_wl4);  // swap WL_GR and WL_next; Line 16 of Algorithm 6
  } while (wl3size > 0);
  
  double runtime = timer.stop();
  printf("initial global relabel from sink: %i iters in %.6fs\n", iter, runtime);
  return iter;
}

static double GPUmaxflow(const ECLgraph g, const ECLgraph d_g, const int source, const int sink, int* const flow, int* const cap, 
                          const int* const d_rnindex, const int* const d_rnlist, const int* const d_retoe)
{
  int* d_flow;  // current flow in each (forward) edge
  if (cudaSuccess != cudaMalloc((void **)&d_flow, g.edges * sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_flow\n");
  
  int* d_cap;  // capacity of each (forward) edge
  if (cudaSuccess != cudaMalloc((void **)&d_cap, g.edges * sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_cap\n");
  if (cudaSuccess != cudaMemcpy(d_cap, cap, g.edges * sizeof(int), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of capacity to device failed\n");
  
  excess_t* d_excess;  // extra flow pooled in node, incoming flow - outgoing flow
  if (cudaSuccess != cudaMalloc((void **)&d_excess, g.nodes * sizeof(excess_t))) fprintf(stderr, "ERROR: could not allocate d_excess\n");
  
  int* d_height;  // estimate of distance to destination (source or sink)
  if (cudaSuccess != cudaMalloc((void **)&d_height, g.nodes * sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_height\n");
  
  int* d_wl1;
  if (cudaSuccess != cudaMalloc((void **)&d_wl1, g.nodes * sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl1\n");
  int wl1size = 0;
  int* d_wl2;
  if (cudaSuccess != cudaMalloc((void **)&d_wl2, g.nodes * sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl2\n");
  int* d_wl2size;
  if (cudaSuccess != cudaMalloc((void **)&d_wl2size, sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl2size\n");
  
  int* d_wl3;
  if (cudaSuccess != cudaMalloc((void **)&d_wl3, g.nodes * sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_wl3\n");
  
  int* d_time;  // tracks the most recent iteration a node was added to a worklist
  if (cudaSuccess != cudaMalloc((void **)&d_time, g.nodes * sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_time\n");
  
  int* d_count;
  if (cudaSuccess != cudaMalloc((void **)&d_count, sizeof(int))) fprintf(stderr, "ERROR: could not allocate d_count\n");
  
  // initialize d_flow, d_excess, d_time, d_height
  int blocks = (std::max(d_g.nodes, d_g.edges) + ThreadsPerBlock - 1) / ThreadsPerBlock;
  init<<<blocks, ThreadsPerBlock>>>(d_g, source, sink, d_flow, d_excess, d_time, d_height);
  cudaMemset(d_wl2size, 0, sizeof(int));
  cudaMemset(d_count, 0, sizeof(int));
  
  // set Global Relabel frequency (number of push-relabel iterations between calls); Line 1 of Algorithm 4
  const double avgDeg = g.edges / g.nodes;
  const int GR_frequency = std::max(100, (int)((g.nodes / 1000) / avgDeg));
  printf("Global relabeling frequency: %i\n", GR_frequency);
  
  GPUTimer timer;
  timer.start();
  
  // perform initial saturating pushes from source; Line 3 of Algorithm 4
  blocks = ((g.nindex[source + 1] - g.nindex[source]) + ThreadsPerBlock - 1) / ThreadsPerBlock;
  initialPush<<<blocks, ThreadsPerBlock>>>(d_g, source, sink, d_excess, d_flow, d_cap, d_wl1, d_wl2size);
  if (cudaSuccess != cudaMemcpy(&wl1size, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of wl2size from device failed\n");
  
  // set initial height values for Phase 1
  const int gr_iters = launchInitialBFS(d_g, source, sink, d_height, d_flow, d_cap, d_rnindex, d_rnlist, d_retoe, d_wl3, d_wl2, d_wl2size, d_time);
  
  // start from Line 6 of Algorithm 4
  launchPushRelabel(d_g, source, sink, d_height, d_excess, d_flow, d_cap, d_rnindex, d_rnlist, d_retoe, 
                    d_wl1, wl1size, d_wl2, d_wl2size, d_wl3, d_time, gr_iters, d_count, GR_frequency);
  
  double runtime = timer.stop();
  
  printf("runtime: %.6fs\n", runtime);
  excess_t result = -1;
  if (cudaSuccess != cudaMemcpy(&result, d_excess + sink, sizeof(excess_t), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of result from device failed\n");
  g_last_maxflow_result = result;
  
  printf("Maximum flow from nodes %i to %i: %li\n", source, sink, (long)result);
  
  if (cudaSuccess != cudaMemcpy(flow, d_flow, g.edges * sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of flow from device failed\n");
  
  // check solution
  cudaMemset(d_wl2size, 0, sizeof(int));
  const int node_blocks = (d_g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock;
  phase2_buildWL<<<node_blocks, ThreadsPerBlock>>>(d_g, source, sink, d_excess, d_wl1, d_wl2size);
  if (cudaSuccess != cudaMemcpy(&wl1size, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of d_wl2size from device failed\n");
  int source_excess;
  if (cudaSuccess != cudaMemcpy(&source_excess, d_excess + source, sizeof(excess_t), cudaMemcpyDeviceToHost)) fprintf(stderr, "ERROR: copying of excess[source] from device failed\n");
  if ((wl1size == 0) && (-source_excess == result)) {
    printf("Verified valid final flow state, no excess nodes remaining, and same absolute excess in source and sink.\n\n");
  } else {
    fprintf(stderr, "ERROR: DID NOT RETURN ALL FLOW TO SOURCE! %i excess nodes remaining, %i source excess, %i sink excess\n", wl1size, source_excess, result);
  }
  
  cudaFree(d_flow);
  cudaFree(d_cap);
  cudaFree(d_excess);
  cudaFree(d_height);
  cudaFree(d_wl1);
  cudaFree(d_wl2);
  cudaFree(d_wl2size);
  cudaFree(d_wl3);
  cudaFree(d_time);
  cudaFree(d_count);
  return runtime;
}
