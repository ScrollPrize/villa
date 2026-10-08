# Shared loading and real-update startup calibration

Always use the optimized parallel sample queue for train and test, managed or
unmanaged. Interleave test jobs in execution order using the same worker facilities.
Keep the broker for ownership/distribution, not a second loading implementation.

Start real training with exclusive, non-pausable/non-adaptive manager ownership;
measure the real initial evaluation and real training updates at different physical
microbatch sizes, without changing logical macro-batches or discarding updates.
Save conservative measured memory estimates, then change to adaptive/pausable use.
No separate calibration command or manual manager configuration is required.
