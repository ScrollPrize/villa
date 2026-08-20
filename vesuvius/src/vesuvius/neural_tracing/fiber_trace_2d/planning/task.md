# Task: show live cached replay preprocessing progress

The default `vc_fiberlets fiberlet-replay` progress display must keep elapsed
time updating while cache work is in progress and must show a useful estimated
fraction and ETA before the fiberlet graph evaluator advances along the
reference. This must work for both newly generated and already persisted cache
chunks and retain the single concise progress line.
