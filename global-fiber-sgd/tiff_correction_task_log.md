# TIFF-only reference visualization correction

- User requires TIFF, not PNG, including inspection crops. Independent plan
  review approved narrow conversion with reopen/verify-before-unlink safeguards.
- Preview exporter now writes LZW TIFF. Approved runner supports maintenance
  conversion of only the seven known experiment detail PNG paths. No projection,
  inference or full-map rerender needed. Existing differing TIFFs not overwritten.
- Eight focused tests passed, including end-to-end TIFF-only export and exact
  conversion/compression/cleanup, preserving unrelated PNGs.
- Run: approved run_classic_fiber_evaluation.py crop-benchmark --config
  /home/hendrik/villa2/global-fiber-sgd/convert_reference_details_tiff.json.
- Prior comparison task log archived as reference_comparison_task_log.md.
- Conversion completed exit0: all seven PNG crops replaced with pixel-identical
  LZW TIFFs; PNG copies removed. Pixel data retained in TIFFs. Full maps unchanged.
