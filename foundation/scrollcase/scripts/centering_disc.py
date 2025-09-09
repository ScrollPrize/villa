"""Generate a centering disc.

Generates a centering disc with the same mount base as a scroll case.
The disc is mounted to the interface plate at the beginning of a scan session to center the tomography stage before acquisition begins.
It guarantees rotational centering and vertical positioning for the scrolls that follow.
At the center of the disc is a cone.
The top of the cone should be centered rotationally within the FOV.
The top of the cone is also positioned to match the bottom of the scroll lining interiors for the cases (i.e. this is the bottom position from which acquisition should proceed upward).
A cone was selected in place of a pin since it is easier to manufacture a cone that is placed at the correct height and is not brittle or easily bent.
"""

from pathlib import Path

from build123d import *
import scrollcase as sc
from ocp_vscode import *
from meshlib import mrmeshpy as mm

NO_SCROLL = 0


def build_disc():
    case = sc.case.ScrollCaseConfig(
        scroll_height_mm=NO_SCROLL, scroll_radius_mm=NO_SCROLL
    )

    # Base
    with BuildPart() as centering_disc:
        add(sc.case.mount_disc(case))
        add(sc.case.bottom_cap(case, with_bolt_protrusions=False))

        # Centering cone
        Cone(
            bottom_radius=case.mount_disc_top_to_lining_bottom,
            top_radius=0,
            height=case.mount_disc_top_to_lining_bottom,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )

    show(centering_disc, reset_camera=Camera.KEEP)

    return centering_disc


disc = build_disc()

# Convert to mesh
disc_mesh = sc.mesh.brep_to_mesh(disc.solids()[0])

mm.saveMesh(disc_mesh, Path("disc.stl"))
