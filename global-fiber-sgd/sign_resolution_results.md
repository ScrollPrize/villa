# Full-volume fiber normal sign results

512,698 whole fibers; 1,320,610 perpendicular events.
10,000 full-batch Adam steps, lr=0.05, weak prior=0.001.

Disagreement is dot-strength-weighted sign opposition—not anatomical error. Broken means model-selected exclusion.
All-event disagreement includes excluded edges; retained disagreement uses only edges whose two fibers remain enabled.

| Method | Broken fibers | Retained events | All-event disagreement | Retained disagreement | Solve time |
|---|---:|---:|---:|---:|---:|
|baked_plus|0.0000%|100.0000%|33.987553%|33.987553%|—|
|radial|0.0000%|100.0000%|1.463706%|1.463706%|—|
|Adam penalty 0.01|9.2253%|49.7687%|0.342420%|0.000013%|22.21s|
|Adam penalty 0.1|0.4685%|98.4322%|0.341368%|0.001069%|22.27s|
|Adam penalty 1|0.0895%|99.5532%|0.320158%|0.161454%|23.40s|

GPU: NVIDIA H100 80GB HBM3; peak tensor allocation 141.69MiB.
One managed4GiB allocation; GPU was shared with training, so timings are observed runs, not isolated throughput benchmarks.
No graph splitting or winding optimization. Source database unchanged.

| Penalty | Broken fibers | Retained events | Opposed retained events |
|---|---:|---:|---:|
|0.01|47,298|657,250|9|
|0.1|2,402|1,299,906|137|
|1|459|1,314,710|2,499|

Artifacts: `global-fiber-sgd/data/sign_resolution_v2/adam_sweep`.
Independent NumPy readout validated every saved solution against the logged hard-readout metrics.
See [model, schema and exact commands](sign_resolution.md).
