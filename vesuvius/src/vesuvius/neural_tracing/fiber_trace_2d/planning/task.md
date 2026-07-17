# Native 3D Trace2CP Beam Lookahead

Extend native 3D Trace2CP beam search so pruning can happen after a short
brute-force future expansion instead of after every single step. The goal is to
survive difficult regions where one or more locally suboptimal steps are needed
before the trace returns to a better path. Start with a default lookahead depth
of `3` steps.
