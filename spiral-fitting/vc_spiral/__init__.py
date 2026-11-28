"""Native extensions used by Spiral fitting."""

# Run from this source tree (as VC3D runs spiral_service.py), this package shadows the installed vc_spiral, which
# holds the compiled modules. Extending the path lets imports of vc_spiral.<module> find them there too
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
