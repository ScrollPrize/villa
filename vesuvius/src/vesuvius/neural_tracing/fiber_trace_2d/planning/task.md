# Task: Non-transitive Mixed-state fiber BP

Change explicit Mixed-state sum-product belief propagation so Mixed represents
uncertain orientation at one fiber rather than an edge label which propagates
to neighboring fibers. A factor touching Mixed must impose no pairwise loss;
instead, each Mixed fiber pays one configurable unary cost. Consistent oriented
neighbors must accumulate evidence for H or V, while conflicting evidence may
make the local Mixed state preferable without encouraging neighboring fibers to
be Mixed.
