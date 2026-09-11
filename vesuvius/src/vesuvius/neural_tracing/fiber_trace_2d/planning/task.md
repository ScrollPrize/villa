# Merge embedding branch and profile with exclusive ownership

Merge fiber-training-embedding into current fiber-training-retrain, preserving both sides' features and all bbox/inference work. Review introduced GPU-manager logical-CPU/lifecycle changes. Ensure regression/performance tests use correct native binding and GPU profiling runs only under an exclusive manager grant. Run the pending tracer/bbox inference profile and report bottleneck.
