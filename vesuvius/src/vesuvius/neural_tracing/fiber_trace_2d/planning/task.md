# Task: retune hard continuation and alignment falloff

Keep edge-local hard split continuity and retune the winding solver around it.

Compare the three normal-alignment confidence families (`none`, `linear`, and
`cosine`) only after independently refining each family's decision-confidence
mode, class weights, finite sign cost, Defect cost, and BP
temperature to a local optimum. Do not compare an untuned falloff variant with
parameters selected for another variant.

Hard continuation and both hard sign classes at a fixed 30-degree alignment
threshold are mandatory in every scenario. Disabled hard signs and alternative
thresholds are outside the valid search space.

Use only the 1024 crop for this tuning pass. Defer the overlapping 2048
larger-context validation until explicitly requested. Report
convergence, exact reference windings, reference-constraint accuracy,
active/Defect counts, continuation and aggregate infringement, solve time, and
total wall time. Do not promote a new default until the tuned comparison is
available. Do not reward a setting merely for disabling difficult reference
constraints.
