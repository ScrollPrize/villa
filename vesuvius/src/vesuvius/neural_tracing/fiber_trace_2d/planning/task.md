# Task: robust sampled-direction anchor refinement

Replace angular line search in local anchor fitting with robust iterative
aggregation of the network's sampled fiber directions. Previous component
directions may condition sample assignment and outlier detection, but every
updated direction must be calculated directly from currently assigned sampled
directions.

Support up to two potentially close directions in a cell. Use deterministic
competitive spatial/directional assignment, detect uncertain or blended
direction outliers adaptively, and trim at most 20% of each component's
presence/spatial evidence mass. Coherent one-direction components with no
detected outliers must retain all evidence.

Remove the existing pre-refinement 10-degree merge so supported close modes
reach robust competition; ordinary downstream NMS remains responsible for
true duplicate anchors.

Apply line-search fractions only to transverse spatial position. Stop after
testing a displacement at or below the existing 0.5-prediction-voxel peak-grid
spacing; leave finer position fitting to the subsequent direction-conditioned
peak refinement. Components whose retained samples do not define a unique
direction must be removed rather than assigned stale state.

Measure performance and compare anchor/fiberlet populations, geometry,
downstream replay failures, and repeatability. Exact numeric or artifact
identity is not required; the user will inspect visual quality after the
measured implementation is available.
