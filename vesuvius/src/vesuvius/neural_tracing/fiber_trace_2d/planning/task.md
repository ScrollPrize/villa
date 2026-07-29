# Task: Bound BLAS Threads in Pyramid Workers

Keep the existing automatic pyramid process count, including 128 processes on
the current node, while forcing every pyramid worker to use one BLAS/OpenMP
thread. The same Fiber inference command must no longer exhaust the process
limit through processes multiplied by OpenBLAS threads.
