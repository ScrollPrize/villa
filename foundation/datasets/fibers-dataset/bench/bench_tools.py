import argparse
import time
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

def run_benchmark(sizes, tiled=False, skip_cpu=False):
    print("=" * 60)
    print(f"{'Volume Size':<15} | {'Filter':<10} | {'Backend':<13} | {'Time (s)':<10} | {'Status':<15}")
    print("-" * 60)
    
    for size in sizes:
        shape = (size, size, size)
        
        np.random.seed(42)
        volume_np = np.random.rand(*shape).astype(np.float32)
        
        for filter_name in ['vesselness', 'ridges']:
            filter_fn = tools.detect_vesselness if filter_name == 'vesselness' else tools.detect_ridges
            tiled_fn = tools.detect_vesselness_tiled if filter_name == 'vesselness' else tools.detect_ridges_tiled

            cpu_time = -1
            if not skip_cpu:
                start_time = time.time()
                try:
                    _ = filter_fn(volume_np)
                    cpu_time = time.time() - start_time
                    print(f"{str(shape):<15} | {filter_name:<10} | {'CPU (NumPy)':<13} | {cpu_time:<10.2f} | {'OK':<15}")
                except Exception as e:
                    print(f"{str(shape):<15} | {filter_name:<10} | {'CPU (NumPy)':<13} | {'N/A':<10} | FAILED: {str(e)[:20]}")
            else:
                print(f"{str(shape):<15} | {filter_name:<10} | {'CPU (NumPy)':<13} | {'SKIPPED':<10} | {'--skip-cpu':<15}")
                
            if HAS_CUPY:
                volume_cp = cp.array(volume_np)
                # Warmup
                try:
                    _ = filter_fn(cp.random.rand(32, 32, 32).astype(np.float32))
                    cp.cuda.Stream.null.synchronize()
                except:
                    pass
                    
                start_time = time.time()
                try:
                    if tiled:
                        _ = tiled_fn(volume_cp, block_size=128, halo=16)
                    else:
                        _ = filter_fn(volume_cp)
                    cp.cuda.Stream.null.synchronize()
                    gpu_time = time.time() - start_time
                    
                    speedup_str = f"{cpu_time / gpu_time:.1f}x" if cpu_time > 0 else "N/A"
                    mode = "GPU (Tiled)" if tiled else "GPU (CuPy)"
                    print(f"{str(shape):<15} | {filter_name:<10} | {mode:<13} | {gpu_time:<10.2f} | OK ({speedup_str} speedup)")
                    
                    mempool = cp.get_default_memory_pool()
                    used_gb = mempool.used_bytes() / (1024**3)
                    print(f"  -> GPU Memory Used: {used_gb:.2f} GB")
                    mempool.free_all_blocks()
                except cp.cuda.memory.OutOfMemoryError as e:
                    mode = "GPU (Tiled)" if tiled else "GPU (CuPy)"
                    print(f"{str(shape):<15} | {filter_name:<10} | {mode:<13} | {'OOM':<10} | FAILED")
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception as e:
                    mode = "GPU (Tiled)" if tiled else "GPU (CuPy)"
                    print(f"{str(shape):<15} | {filter_name:<10} | {mode:<13} | {'N/A':<10} | FAILED: {str(e)[:20]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark fibers-dataset tools.")
    parser.add_argument('--sizes', nargs='+', type=int, default=[64, 128, 256], help='Volume cube sizes to test')
    parser.add_argument('--tiled', action='store_true', help='Test tiled execution to save memory')
    parser.add_argument('--skip-cpu', action='store_true', help='Skip CPU benchmarking (useful for large sizes)')
    args = parser.parse_args()
    
    run_benchmark(args.sizes, args.tiled, args.skip_cpu)
