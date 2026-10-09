# H-fiber direction alignment results

Stage3 after inward normals and fixed H/V. No new rejection, cutting, or global U/V solve.
All numbers concern retained H–H parallel events; disagreement is model consistency, not anatomical error.

| Method | Disagreeing events | Event disagreement | Weighted disagreement |
|---|---:|---:|---:|
|Original stored point order|444,132|12.325452%|12.325248%|
|Umbilicus only|25,114|0.696958%|0.698138%|
|Aligned H directions|1,721|0.047761%|0.047735%|

Retained H fibers: 251,658; parallel events: 3,603,373; new rejections:0.
Changed directions vs umbilicus-only: 1,612 (0.640552%).
Uncertain probability in(0.1,0.9): 0.000000%. This is optimizer certainty, not calibrated confidence.
Graph: 82,250 components, including 58,793 isolated H fibers.
Weak final circulation evidence: 1,217 components; independent gauges remain a limitation.
Preparation: 84.12s; 10000full-batch Adam steps: 25.80s on NVIDIA H100 80GB HBM3.
Peak CUDA tensors: 274.65MiB. Shared managed GPU, one run, not an isolated speed study.

Output: `/home/hendrik/villa2/global-fiber-sgd/data/h_direction_v1/solve/solution.npz`.
CW directions are exact negatives of CCW; probabilities complemented and logits negated. Pair agreement is identical.
Global assignment remains paused. Its prior experimental runs did not consume this new stage.
