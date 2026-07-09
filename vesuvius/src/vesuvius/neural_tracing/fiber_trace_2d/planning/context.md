name                    total loader_total   descriptor  line_window lasagna_norm strip_coords coord_augmen volume_sampl value_augmen  line_coords        to_u8      overlay
  unaugmented            2959.5       2956.0          0.1          0.2         39.6        470.2         61.3       2182.0         27.7         82.5          0.1          3.3
  shift_x_min             563.9        563.5          0.2          0.1         37.7        460.5          2.4          1.7          0.3         60.4          0.1          0.2
  shift_y_min             596.0        595.7          0.2          0.1         40.8        480.9          2.2          2.4          0.4         68.5          0.1          0.3
  ok this is the timing - first is slow (which is fine, filling cache), but the others are pretty slow on:
  - strip coords
  - line coors
  my suspicion is this is due to python for loops? please report (no coding)


• Yes, that timing strongly points to Python-side loops and repeated per-augmentation recomputation.

  Likely breakdown:

  - strip_coords ~460-480 ms: this is probably building the VC3D-style strip coordinate field in Python/Numpy per contact-sheet cell. If it walks offsets / line samples / strip columns in 
    Python, that cost will dominate even when volume sampling is cached and fast.

  - line_coords ~60-80 ms: same pattern, but smaller. It likely recomputes the transformed fiber-line overlay coordinates for every augmentation cell, also using Python loops over line    
    samples.

  - lasagna_norm ~38-40 ms: also repeated per cell, likely resampling/decoding the same local normals each time. This is not as bad, but it should probably be computed once per selected   
    CP/base patch and reused across augment variants.

  The key issue is that the contact sheet currently treats every augmentation cell like an independent sample build. But for one CP, most expensive geometry is shared:

  - local line window
  - Lasagna normals along the local window
  - base strip coordinate scaffold
  - base line coordinates
  - maybe oversized coordinate support

  Only the augmentation parameters should differ per cell.

  So yes: very likely not VC3D chunk I/O anymore. After the first volume_sample, cached sampling is ~1-2 ms, while geometry construction stays ~500 ms every time. That suggests the next   
  fix should be to cache/reuse the CP-local geometry once per export/sample, then apply only per-cell coordinate transforms before calling the sampler.

