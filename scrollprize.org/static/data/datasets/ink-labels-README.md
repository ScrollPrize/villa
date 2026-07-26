# `ink-labels`

Binary ink masks marking the location of ink on papyrus surfaces, for training
and evaluating ink-detection models.

An ink label is a binary image: each pixel marks whether ink is present at the
corresponding location on a sampled surface. For detached fragments, labels are
produced by aligning an infrared photo of the exposed writing to the surface
volume and manually tracing the ink. For scrolls — where no infrared ground
truth exists — labels begin as hand-annotated ink strokes and are refined
through iterative pseudo-labeling; models trained on these labels can then
detect ink elsewhere in the scroll.

## Layout

The dataset is organized by scroll, then by segment:

```
<scroll>/<segment>/...
```

Top-level scroll/collection folders:

| Folder | Notes |
| --- | --- |
| `phercparis4/` | Scroll 1 (PHercParis4). Segments named `w00_…`, `w01_…`, etc. |
| `0009b/`, `0139/`, `0500p2/`, `1667/`, `814/`, `841/` | Additional scroll / segment collections. |
| `man5/` | Manually labeled set. |
| `unused/` | Segments retained but not used in current training. |

Within each segment you will typically find a sampled **surface volume**
(an image stack of the papyrus surface) and one or more **binary ink label**
images aligned to it.

**Total:** ~804 GB across ~2.38M files.

## License

See <https://dl.ash2txt.org/LICENSE.txt>.
