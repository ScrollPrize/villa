# FM-P03 — Surface wrapper silently changes numbered selectors

Older omission present at `0c7e4e521`. `vc_object_store/python/vc_object_store/surfaces.py::read_surface` converts non-null `version` with `int(version)` before the native session boundary. A fractional version is silently interpreted as a different exact numbered version. This is not a complete-reference codec problem or a request to remove explicit numbered discovery.

Runtime probe: create surface version 2, call `read_surface(..., version=2.5)`, and receive the version-2 reference/geometry successfully. Booleans/numeric strings are similarly admitted by this wrapper's cast. `vesuvius/neural_tracing/inference/surface_store.py::read_surface` delegates to this wrapper; complete references bypass numeric selection and are unaffected.

Let the shared native selector boundary validate the original value rather than pre-coercing it in Python. Do not add a second local validation table. Verify native/session/wrapper parity for valid integers, malformed fractions/bools/strings and full exact references. Source/runtime confirmed by root; the additional independent Python review lane did not finish because the reviewer hit its usage limit.
