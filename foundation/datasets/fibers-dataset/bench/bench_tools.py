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

def detect_vesselness_tiled(volume_cp, block_size=128, halo=16):
    """
    Run detect_vesselness in blocks to fit in GPU memory.
    """
    Z, Y, X = volume_cp.shape
    result = cp.zeros((Z, Y, X), dtype=volume_cp.dtype)
    
    for z in range(0, Z, block_size):
        for y in range(0, Y, block_size):
            for x in range(0, X, block_size):
                # Calculate coordinates with halo
                z0, z1 = max(0, z - halo), min(Z, z + block_size + halo)
                y0, y1 = max(0, y - halo), min(Y, y + block_size + halo)
                x0, x1 = max(0, x - halo), min(X, x + block_size + halo)
                
                # Extract block
                block = volume_cp[z0:z1, y0:y1, x0:x1]
                
                # Process block
                block_res = tools.detect_vesselness(block)
                
                # Crop halo and write back
                bz0 = z - z0
                bz1 = bz0 + min(block_size, Z - z)
                by0 = y - y0
                by1 = by0 + min(block_size, Y - y)
                bx0 = x - x0
                bx1 = bx0 + min(block_size, X - x)
                
                result[z:z+bz1-bz0, y:y+by1-by0, x:x+bx1-bx0] = block_res[bz0:bz1, by0:by1, bx0:bx1]
                
                # Free memory aggressively inside loop
                del block
                del block_res
                cp.get_default_memory_pool().free_all_blocks()
                
    return result

def run_benchmark(sizes, tiled=False, skip_cpu=False):
    print("=" * 60)
    print(f"{'Volume Size':<15} | {'Backend':<13} | {'Time (s)':<10} | {'Status':<15}")
    print("-" * 60)
    
    for size in sizes:
        shape = (size, size, size)
        
        np.random.seed(42)
        volume_np = np.random.rand(*shape).astype(np.float32)
        
        cpu_time = -1
        if not skip_cpu:
            start_time = time.time()
            try:
                _ = tools.detect_vesselness(volume_np)
                cpu_time = time.time() - start_time
                print(f"{str(shape):<15} | {'CPU (NumPy)':<13} | {cpu_time:<10.2f} | {'OK':<15}")
            except Exception as e:
                print(f"{str(shape):<15} | {'CPU (NumPy)':<13} | {'N/A':<10} | FAILED: {str(e)[:20]}")
        else:
            print(f"{str(shape):<15} | {'CPU (NumPy)':<13} | {'SKIPPED':<10} | {'--skip-cpu':<15}")
            
        if HAS_CUPY:
            volume_cp = cp.array(volume_np)
            try:
                _ = tools.detect_vesselness(cp.random.rand(32, 32, 32).astype(np.float32))
                cp.cuda.Stream.null.synchronize()
            except:
                pass
                
            start_time = time.time()
            try:
                if tiled:
                    _ = detect_vesselness_tiled(volume_cp, block_size=128, halo=16)
                else:
                    _ = tools.detect_vesselness(volume_cp)
                cp.cuda.Stream.null.synchronize()
                gpu_time = time.time() - start_time
                
                speedup_str = f"{cpu_time / gpu_time:.1f}x" if cpu_time > 0 else "N/A"
                mode = "GPU (Tiled)" if tiled else "GPU (CuPy)"
                print(f"{str(shape):<15} | {mode:<13} | {gpu_time:<10.2f} | OK ({speedup_str} speedup)")
                
                mempool = cp.get_default_memory_pool()
                used_gb = mempool.used_bytes() / (1024**3)
                print(f"  -> GPU Memory Used: {used_gb:.2f} GB")
                mempool.free_all_blocks()
            except cp.cuda.memory.OutOfMemoryError as e:
                mode = "GPU (Tiled)" if tiled else "GPU (CuPy)"
                print(f"{str(shape):<15} | {mode:<13} | {'OOM':<10} | FAILED")
                cp.get_default_memory_pool().free_all_blocks()
            except Exception as e:
                mode = "GPU (Tiled)" if tiled else "GPU (CuPy)"
                print(f"{str(shape):<15} | {mode:<13} | {'N/A':<10} | FAILED: {str(e)[:20]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark fibers-dataset tools.")
    parser.add_argument('--sizes', nargs='+', type=int, default=[64, 128, 256], help='Volume cube sizes to test')
    parser.add_argument('--tiled', action='store_true', help='Test tiled execution to save memory')
    parser.add_argument('--skip-cpu', action='store_true', help='Skip CPU benchmarking (useful for large sizes)')
    args = parser.parse_args()
    
    run_benchmark(args.sizes, args.tiled, args.skip_cpu)
