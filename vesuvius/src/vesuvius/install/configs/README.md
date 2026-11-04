# Annotated cube inventory

`cubes.yaml` maps scroll IDs, scan energies, resolutions and cube origins to
directories containing paired `<origin>_volume.nrrd` and `<origin>_mask.nrrd`
files. Masks contain segmentation instance IDs, not ink labels.

```yaml
scroll_id:
  energy:
    resolution:
      "ZZZZZ_YYYYY_XXXXX": "https://host/path/to/cube/"
```

The inventory ships with the package and is read by `vesuvius.list_cubes()` and
`vesuvius.Cube`. Quote origin keys to preserve them as strings. Discovery is
local and does not download data or refresh this file.

Scroll volumes and segments use the `vc_reader` Open Data catalog instead.
