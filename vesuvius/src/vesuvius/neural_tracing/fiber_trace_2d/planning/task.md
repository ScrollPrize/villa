# Trace2CP Last-Point Similarity Columns

Update the forward/reverse last-similarity debug images in single-pair
Trace2CP visualization.

The debug maps should show the similarity evidence used along the trace rather
than a single full-image map against the final trace point:

- For each newly placed trace point, sample the embedding at the previous
  accepted trace point.
- Compute the cosine similarity of the columns around the newly placed point
  against that previous-point embedding.
- Use half the configured trace step length, rounded up, as the column-band
  radius.
- Paint those columns into the forward or reverse debug map. Small overwrites
  are acceptable.
- Keep the start CP, target CP, and global CP-bank similarity panels unchanged.
