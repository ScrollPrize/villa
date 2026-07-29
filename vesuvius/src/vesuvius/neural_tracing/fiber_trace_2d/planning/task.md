# Native C++ Trace2CP Parity Fix Plan

The native C++ fiber tracer is close to the Python Trace2CP baseline but still
produces more whole-fiber restarts on the current S1 persisted inference
manifest: 7 restarts versus the expected Python baseline of about 3. The raw
persisted-output quantization/compression difference is explicitly out of scope
for this task. All other implementation differences found in the native C++
port should be removed so the C++ tracer follows the existing native Trace2CP
spec.

The fixes should address:

- circular candidate-cone generation instead of the accidental square fan
- Python/spec-compatible branch selection after the start point
- Python/spec-compatible angle-squared smoothness with free-angle handling
- cumulative tangent smoothness
- target-plane crossing interpolation and continuation from the crossing point
- beam spatial-diversity pruning
- CLI/config/default exposure for the missing native Trace2CP controls
- regression tests and a benchmark comparison on the user-provided command

Do not change the persisted inference representation as part of this task.
