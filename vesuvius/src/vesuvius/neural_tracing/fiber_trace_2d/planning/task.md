# Native 3D Trace2CP Vectorized Beam Lookahead

Plan how to vectorize native 3D Trace2CP beam lookahead as much as possible on
GPU. Current candidate scoring is tensorized per beam state, but beam frontier
expansion, lookahead depth handling, and child bookkeeping still loop in Python.
The next implementation should batch all active beam/frontier states for each
lookahead depth, generate candidates on GPU, score them in one call, and prune
with torch operations before converting the selected final path back to Python
objects for reporting/visualization.
