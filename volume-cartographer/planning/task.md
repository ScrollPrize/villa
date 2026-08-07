# Task

Allow the synthetic rendering regression gate to run across Valgrind version
updates. Continue recording the profiler version for diagnostics, but do not
fail solely because it differs from the frozen reference; let the modeled
runtime tolerance detect material profiler or renderer changes.
