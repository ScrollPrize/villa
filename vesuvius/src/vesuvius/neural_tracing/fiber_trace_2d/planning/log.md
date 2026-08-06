# initial work
- vc3d based (imported c++) strip sampling with full cache & remote support
- initial loader was ~1 patch/second
- fast coord and image based augmentations and vectorized coord generation -> 100 patches/s loading speed
- point based gt is transformed same way
- shallow 5-layer cnn, now 10 layer resnet
- ambiguous lasagna style dir estimation only for now

# initial results
- learns very fast (few k its for 5layer resnet)
- already seems better/on-par to gt -> need gt refinement/self-supervision
- at 5 layers receptive field is a bit small (scaledown 1) so switched to 10 layer

# first tracing use
- trace out step-by-step
- support TTA to get a flock of traces
- med-tta tracing - take the median in-loop of the TTA traces

# speed opt
- loading now does 460patches/s with some caching - 240 in actual training (with deeper net!) - but only 100 when doing 1 patch/cp (with idle cpu available ...)

# various
- trace2cp metric now goes both ways
- finding: group-norm with single supervision points does cause us to buidl a per-patch classifiere instead of a cnn inference (compare dir_vis_global.jpg)
- perf-opt: llms are just bad at it if its not just micro-tuning and paramater fiddling - we now get 185patchs/s with 1patch/cp - good enough for now
- added embedding and constrastive learning & tracing
- observation: embedding degnerates to become sheet recognition?
- removed shear and scale: be28aeadd2145db9691fa38b0deb4c35ffa41d69 
- direct img as search cand - did not work
- in all cases so far: embedding only seems to choose between point is on-fiber or off-fiber (e.g. embedding inverts or not)
- looser embedding also does not work
- presence does
- dp with adjusted weights looks correct apart from z-offset time PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_2d.runner $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --export-dir ./ --checkpoint $VES/data/fiber_trace_2d_runs/top_20260713_004411/snapshots/best.pt --trace2cp-vis --sample-index 12 --trace2cp-combined --trace2cp-use-presence --line-trace-step 1 --trace2cp-z-max-layer 40 --trace2cp-z-step-voxels 0.5 --trace2cp-dp  --trace2cp-z-search  --trace2cp-combined-direction-weight 0.5
- fixed side-view normal being identical to side-view y axis now:
time PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_2d.runner $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --export-dir ./ --checkpoint $VES/data/fiber_trace_2d_runs/top_20260713_004411/snapshots/best.pt --trace2cp-vis --sample-index 12 --trace2cp-combined --trace2cp-use-presence --line-trace-step 1 --trace2cp-z-max-layer 40 --trace2cp-z-step-voxels 0.5 --trace2cp-dp  --trace2cp-z-search  --trace2cp-combined-direction-weight 0.5
- gives nearly perfect results!

-time PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_3d.trace2cp_tool $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_all_64_sd2.json --checkpoint $VES/data/fiber_trace_3d_runs/s1a_nml_all_3d_64_2_multidir_b4_20260717_121412/snapshots/best.pt --export-dir ./ --fiber-json $VES/data/train_fibers/fibers_test_paul_4/kb_202606*001.json --beam-lookahead-steps 1 --beam-width 8 --smoothness-normal-weight 0.1 --smoothness-tangent-weight 100.0 --sample-index 13
on 4e0de37ac02817835eab6b903ea4eb211b560015 is very good
