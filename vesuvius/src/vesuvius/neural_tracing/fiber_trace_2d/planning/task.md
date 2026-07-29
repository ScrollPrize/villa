# Native VC3D 3D Fiber Tracer

Port the current Python 3D fiber Trace2CP tracer into native
volume-cartographer C++.

The native path should use the precomputed 3D fiber inference volume produced
by the fiber inference preprocessing, not run neural inference inside VC3D.
Volume access, local/remote handling, chunk caching, and metadata should reuse
the existing VC3D/Lasagna volume facilities.

The integration has three parts:

- volume access with local and remote support, reusing existing VC3D facilities
  rather than introducing a new volume/cache path;
- a separate built-in C++ fiber tracer library that implements the Trace2CP
  features used by the current Python reference command, minus visualization,
  and outputs an updated fiber/line annotation;
- GUI integration in the line annotation window.

Initial scope is CP-to-CP segment tracing only, with progress output.

The CP-to-CP tracer should:

- run the same bidirectional search used by the Python reference;
- use the precomputed fiber inference volume for direction/presence fields;
- use Lasagna normals for tangent/normal smoothness terms;
- keep the original CP coordinates untouched in all cases;
- reject applying an optimization if either endpoint error exceeds 50 um;
- if endpoint errors are below the threshold, fuse the bidirectional traces
  with the same center-fusion-lerp behavior as the Python reference and replace
  only the line points between the two CPs.

For the first GUI test, expose this as a Ctrl-right-click segment action in the
line annotation window.

Successful segment updates should be marked as tracer-optimized with the
maximum endpoint error recorded. Regular line annotation re-optimization must
not re-optimize unchanged tracer-optimized segments. Moving a CP, deleting a CP,
or adding a CP inside such a segment must discard the tracer-optimized flag and
error metadata and restore regular line annotation behavior for that segment.

While native tracing is running, block other line modifications and show
progress in the UI.

Fiber inference datasets should be stored in the project in the same way that
Lasagna datasets are stored, including local/remote manifest information needed
to reopen them later.
