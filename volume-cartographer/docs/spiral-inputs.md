# Spiral input list

The Spiral workspace lists inputs added or modified during the editing session
by default. Applied and committed changes remain visible. Enable **Show original
dataset inputs** to also see unchanged inputs loaded from the dataset. The text
filter applies to both views. Inputs with errors remain visible in the default
view so a failed edit cannot disappear.

Visibility does not change which inputs are checked, used for fitting, or selected
for Apply/Commit. Session activity resets when a new service editing workspace is
created; rebuilding the fit within that workspace does not reset it.

Build from the repository root using the existing configured build directory:

```sh
AGENTS_AGENT_MODE=1 ninja -C volume-cartographer/build -j2 VC3D \
  test_spiral_input_draft test_spiral_input_workflow
ctest --test-dir volume-cartographer/build -R '^spiral_input_draft$' --output-on-failure
QT_QPA_PLATFORM=offscreen \
  SPIRAL_TEST_PYTHON="$PWD/spiral-fitting/.venv/bin/python" \
  volume-cartographer/build/bin/test_spiral_input_workflow
volume-cartographer/build/bin/VC3D
```

The Qt workflow test runs its own temporary local service. It requires permission
to bind loopback sockets. Snapshot implementation and benchmark details are in
the [Spiral fitting README](../../spiral-fitting/README.md).
