# Fresh full-resolution five-scroll training

Stop the current refinement training. Start a new training from scratch without
refinement or embeddings, using the same eligible five-scroll JSON-only data and
training settings as the current regular baseline, except that the U-Net emits
full-resolution output (`model_3d.output_stride: 1`) instead of stride 4.
