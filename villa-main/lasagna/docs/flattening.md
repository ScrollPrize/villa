# Quad Mesh Regularization via Green Strain Tensor

## Overview

For a regular quad mesh stored as:

```python
X.shape == (H, W, 3)
```

we can formulate mesh regularization directly in terms of a local deformation gradient and its induced metric tensor.

This provides a clean continuum-style formulation equivalent to enforcing:

* equal edge lengths,
* orthogonality,
* diagonal consistency,
* shear suppression.

The formulation is closely related to:

* nonlinear elasticity,
* conformal parameterization,
* ARAP,
* symmetric Dirichlet energies,
* Green strain tensors.

---

# Local Quad Differential

For each quad cell, define local tangent vectors:

```python
a = X[1:, :-1] - X[:-1, :-1]
b = X[:-1, 1:] - X[:-1, :-1]
```

Interpretation:

* `a` = vertical lattice direction
* `b` = horizontal lattice direction

These form a discrete deformation gradient:

[
F = [a ;; b]
]

---

# Metric Tensor

Compute the induced metric tensor:

[
G = F^T F
]

whose entries are:

```python
g11 = (a * a).sum(dim=-1)
g22 = (b * b).sum(dim=-1)
g12 = (a * b).sum(dim=-1)
```

So:

[
G =
\begin{bmatrix}
|a|^2 & a \cdot b \
a \cdot b & |b|^2
\end{bmatrix}
]

This tensor captures:

* edge lengths,
* anisotropy,
* shear,
* orthogonality.

---

# Green Strain Tensor

The Green strain tensor is:

[
E = \frac12(F^T F - I)
]

or more generally:

[
E = \frac12(F^T F - h^2 I)
]

where:

* `h` = desired edge length.

Perfect square lattice condition:

[
F^T F = h^2 I
]

which implies:

* equal edge lengths,
* orthogonal directions,
* square local structure.

---

# Fixed Edge-Length + Orthogonality Loss

For target spacing `h`:

```python
h2 = h * h

loss = (
    (g11 - h2)**2 +
    (g22 - h2)**2 +
    2 * g12**2
).mean()
```

This enforces:

* `|a| = h`
* `|b| = h`
* `a · b = 0`

Equivalent to:

* edge constraints,
* diagonal constraints,
* orthogonality constraints.

---

# Explicit Green-Strain Energy

You can explicitly compute strain components:

```python
E11 = 0.5 * (g11 - h2)
E22 = 0.5 * (g22 - h2)
E12 = 0.5 * g12
```

Then optimize:

```python
loss = (
    E11**2 +
    E22**2 +
    2 * E12**2
).mean()
```

This is equivalent to a Saint-Venant–Kirchhoff elasticity energy.

---

# Scale-Invariant Version

Sometimes local scaling should be allowed while preserving square-ness.

Define local isotropic scale:

[
s^2 = \frac12(g11 + g22)
]

Implementation:

```python
s2 = 0.5 * (g11 + g22)

loss = (
    (g11 - s2)**2 +
    (g22 - s2)**2 +
    2 * g12**2
).mean()
```

This penalizes:

* anisotropy,
* shear,

while allowing:

* local uniform scaling.

Equivalent to:

* isotropic conformal regularization.

---

# Relation to Diagonal Constraints

Diagonal constraints are implicitly encoded.

For quad directions `a,b`:

[
|a+b|^2

|a|^2 + |b|^2 + 2a \cdot b
]

If:

* `|a| = |b| = h`
* `a · b = 0`

then:

[
|a+b|^2 = 2h^2
]

Thus diagonal lengths become automatically:

[
\sqrt{2} h
]

No explicit diagonal term is necessary.

---

# Relation to Classical Parameterization Energies

## Symmetric Dirichlet

Minimizes singular value distortion:

[
\sigma_1^2 + \sigma_2^2
+
\sigma_1^{-2} + \sigma_2^{-2}
]

Prefers:

[
\sigma_1 = \sigma_2 = 1
]

Equivalent goal:

* isotropic local metric.

---

## ARAP

As-rigid-as-possible energy:

[
|F - R|^2
]

Prefers local rotations.

---

## Green-Strain Formulation

Directly constrains:

[
F^T F
]

Advantages:

* simple,
* fully differentiable,
* easy in PyTorch,
* quad-aware,
* sparse/local,
* physically interpretable.

---

# Important Limitation: Flips

The metric tensor:

[
F^T F
]

does NOT encode orientation.

Therefore:

* flipped quads,
* inversions,
* fold-overs

are not detected.

Additional constraints/barriers are needed.

---

# Simple Area Barrier

For 3D embedded quads:

```python
n = torch.cross(a, b, dim=-1)
area = torch.norm(n, dim=-1)
```

Possible penalties:

```python
loss_area = (1.0 / (area + eps)).mean()
```

or:

```python
loss_area = torch.relu(min_area - area).pow(2).mean()
```

---

# Conceptual Interpretation

This framework is essentially:

## Discrete Nonlinear Elasticity

The quad mesh behaves like:

* a hyperelastic sheet,
* with preferred metric tensor:

[
G_0 = h^2 I
]

The optimization attempts to keep the local metric close to this ideal square metric.

This connects directly to:

* cloth simulation,
* shell energies,
* harmonic maps,
* conformal parameterization,
* geometry processing,
* continuum mechanics.

---

# Practical Recommendation

A strong practical quad regularizer is:

```python
loss =
    λ_metric * metric_loss +
    λ_area   * area_barrier +
    λ_smooth * smoothness_loss
```

Typical components:

* metric isotropy,
* positive area enforcement,
* normal smoothness,
* Laplacian smoothing,
* boundary constraints.

---

# Minimal Recommended Implementation

```python
a = X[1:, :-1] - X[:-1, :-1]
b = X[:-1, 1:] - X[:-1, :-1]

g11 = (a * a).sum(dim=-1)
g22 = (b * b).sum(dim=-1)
g12 = (a * b).sum(dim=-1)

h2 = h * h

metric_loss = (
    (g11 - h2)**2 +
    (g22 - h2)**2 +
    2 * g12**2
).mean()

n = torch.cross(a, b, dim=-1)
area = torch.norm(n, dim=-1)

area_loss = (1.0 / (area + 1e-6)).mean()

loss = metric_loss + 0.01 * area_loss
```


  1. SLIM/local-global solve instead of Adam
     Symmetric Dirichlet is usually not optimized by plain Adam. A SLIM-style loop alternates between local per-quad projections and a global sparse linear solve. That can converge in tens
     of iterations instead of thousands. This is probably the most “correct” formulation shift.
  2. Optimize a source-surface UV map, then invert it
     Instead of optimizing M[out -> source], assign each valid tifxyz vertex a 2D flattened coordinate U[source_h,source_w,2]. Flatten the source mesh once, then rasterize/invert U onto
     the regular output grid. This turns the problem into standard mesh parameterization and avoids bilinear sampling validity churn during optimization.
  3. Use a diffeomorphic flow field
     Parameterize the map as M = identity + integrate(v) or as a composition of small warp updates. Each update is small and composed, not directly written into M. This can preserve
     topology much better and avoid checkerboard/fold artifacts, but it changes the optimizer structure.
  4. Optimize sparse control points + solve dense map by interpolation
     Instead of every output pixel being a variable, optimize a sparse lattice or landmarks and generate the dense inverse map with B-splines/thin-plate/RBF interpolation. Then
     progressively add control points only where residual is high. Much fewer variables, faster convergence.
  5. Flatten by tangent-field integration
     Estimate local tangent directions/metric on the tifxyz surface, solve a Poisson/integration problem to produce a planar coordinate field, then only use the current loss for cleanup.
     This is closer to “derive a flattening” than “discover it by gradient descent.”
  6. Graph/spectral initialization
     Build a graph over valid source cells, compute a spectral/MDS/Isomap-style 2D embedding from approximate geodesic distances, then fit the regular output map to that. Expensive once,
     but could land much closer than identity for difficult geometry.
  7. Constrained projected Newton / barrier method
     Keep the same variables, but replace Adam with a line-searched second-order method and explicit fold constraints. The key difference is rejecting steps that worsen fold/orientation
     constraints rather than hoping penalties steer Adam back.
  8. Patchwise flatten + stitch
     Flatten overlapping local patches independently, where the problem is nearly linear, then solve a global stitching/Poisson blend. This could be much faster on large surfaces and more
     robust around holes, but it needs careful seam handling.
