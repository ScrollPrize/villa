# Automated Fiber Volume input

Spiral Fit can take an Automated Fiber Volume (`.afv`, see
`volume-cartographer/docs/fiber-collections.md`) as an extra fiber source, next
to the native fiber directory, which is unchanged.

## In VC3D

* The Spiral panel has an optional **Automated Fiber Volume** field below
  **Fibers**. It is filled with the volume open in the Automated Fiber Volume
  dock unless you choose a file yourself, and clearing it removes only this
  source. Setting it enables fiber input.
* A local `.afv` is uploaded to the service once, at Initialize or Rebuild. It
  is checked by SHA-256 and reused across sessions.
* When a Spiral session is active, **Add to Spiral Fit** in the Automated Fiber
  Volume dock sends only the selected fiber, through the existing single-fiber
  path. The automatic suggestion in the field is then dropped.
* The Automated Fiber Volume overlay is also drawn in the Spiral views, at the
  scale of the Spiral preview volume like native fibers, and clicking a trace
  there selects the fiber.

## Loading

The fibers are read directly from the `.afv`, read-only, and go through the
same loader as native fibers: coordinate mapping, control spans, decimation and
H/V classification. Only fibers whose bounds overlap the fit's Z range are
decoded.

In addition to the documented format, Spiral requires:

* `coordinate_base_shape_zyx` in the `frame` metadata: the shape of the CT
  volume the coordinates refer to, as native fibers carry it;
* no cross-fiber branch references in the fiber annotations.

## Tests

```sh
python -m pytest tests/test_afv_input.py tests/test_afv_service.py -q
```

The tests write small `.afv` files themselves.
