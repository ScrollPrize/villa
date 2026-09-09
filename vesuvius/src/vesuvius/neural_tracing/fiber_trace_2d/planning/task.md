# Task: tune Fiber3D gradient clipping and peak learning rate

Implement and run a from-scratch LR-range and gradient-norm experiment using
the historical S1a NML training data/model configuration. Use six idle H100s
(`0,1,4,5,6,7`), excluding occupied GPUs 2 and 3, but reproduce the final
eight-GPU, 16-samples/GPU optimizer batch of exactly 128 samples through
eight full logical rank-local batches and one manual gradient reduction. Infer
a conservative global-L2 clipping policy and peak LR for that final training
run.
