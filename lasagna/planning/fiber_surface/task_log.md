# Future render resolution

- Prior fitting/results archived in tangential_geometry_log.md.
- Defaults and full-render presets now8baseUV/pixel and4px lines,formerly4/8.
- Same UV bounds and CT source scale. About quarter as many output pixels; no
  measured wall-time claim. Same band becomes7351x3188 instead of14701x6376.
- Existing images and saved data/config.json provenance untouched. Overlay-only
  refresh preset remains8px because it targets an old full-resolution image.
- No new solve or render requested. Independent review approved;5renderer tests
  passed including half-resolution registration and4px line-width check.
