# H/V stage2 results

Whole original fibers; stage1 penalty0.1 exclusions fixed. V=axial, H=perpendicular to the umbilicus axis.
Constraint consistency is not anatomical ground truth or proof of a family-switching fiber.
New broken % uses stage1-retained fibers; total broken % uses all original fibers. Event/arc retention uses stage1-retained input.
Entropy is minimized late; hard readout is argmax(H,V,broken). Shared managed GPU timings are not isolated benchmarks.

| Cost / penalty / entropy | New broken | Total broken | Retained arc | Retained events | Parallel errors | Perpendicular errors | All errors | Ambiguous | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Axis-only baseline|0.0000%|0.4685%|100.0000%|100.0000%|5.3393%|13.8457%|6.8587%|100.0000%|—|
|uniform_b0.1_e0|3.6091%|4.0607%|94.8640%|87.8366%|0.0038%|0.4473%|0.0847%|0.4172%|89.68|
|uniform_b0.1_e0.01|3.6685%|4.1198%|94.6814%|87.1512%|0.0037%|0.4394%|0.0835%|0.0002%|88.86|
|uniform_b1_e0|0.4298%|0.8962%|99.1581%|97.2554%|0.4155%|4.8348%|1.2117%|0.3923%|88.93|
|uniform_b1_e0.01|0.4317%|0.8982%|99.1544%|97.2438%|0.4149%|4.8298%|1.2102%|0.0000%|88.89|
|degree_mix_b0.1_e0|4.0876%|4.5370%|95.2663%|90.9978%|0.0027%|0.3146%|0.0580%|0.4047%|88.86|
|degree_mix_b0.1_e0.01|4.0835%|4.5329%|95.2522%|90.9436%|0.0025%|0.3171%|0.0583%|0.0000%|88.85|
|degree_mix_b1_e0|0.1654%|0.6331%|99.7164%|99.1359%|0.6841%|5.7461%|1.5903%|0.3917%|88.85|
|degree_mix_b1_e0.01|0.1656%|0.6333%|99.7160%|99.1347%|0.6839%|5.7463%|1.5902%|0.0000%|88.95|

Inherited broken: 2,402; active fibers: 510,296; active events: 7,277,943.
GPU: NVIDIA H100 80GB HBM3; peak tensor allocation: 662.4MiB.

Saved ternary arrays were audited against input IDs, frozen exclusions and independently recomputed retained/violated events.
