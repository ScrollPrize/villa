# Accelerate Startup Compact Geometry Build

The compact in-RAM fiber-line geometry store is correct and small, but full
dataset construction is too slow: the current `loader_example.json` startup
build took `8m53s` for `464` records.

Implement the next optimization pass using all four agreed directions:

- parallelize compact geometry construction by record;
- vectorize Lasagna normal sampling/decoding per record;
- only preprocess line points that can affect a configured CP source window;
- combine those changes and benchmark the startup build plus load-only hot
  path.

Use the existing `loader_workers` setting for startup geometry parallelism
instead of adding another config key. `loader_workers=1` must remain the
deterministic serial/debug path.
