# Persist Per-Segment Manifest Identities

- Store the Lasagna manifest identity used for Lasagna interpolation and the
  fiber-inference manifest identity used for trace interpolation.
- For open-data catalogue Lasagna, store the public remote manifest URL rather
  than the local cache path.
- Keep unused identities empty for interpolation modes that did not consult
  those datasets.
