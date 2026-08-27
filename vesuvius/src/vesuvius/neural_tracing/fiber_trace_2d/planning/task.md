# Task: Binary sum-product fiber belief propagation

Add an experimental sum-product alternative to the existing binary min-sum
fiber BP. Keep the same final no-split perpendicular factor graph and central
straight H seed, but report normalized approximate `P(H)` marginals rather
than a sigmoid of min-marginal cost gaps. Run it through the BP-only path and
compare its constraint-consistency and Mixed-vs-trusted discrimination with
the committed min-sum result.
