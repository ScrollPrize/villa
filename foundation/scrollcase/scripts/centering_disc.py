"""Generate a centering disc.

Generates a centering disc with the same mount base as a scroll case.
The disc is mounted to the interface plate at the beginning of a scan session to center the tomography stage before acquisition begins.
It guarantees rotational centering and vertical positioning for the scrolls that follow.
At the center of the disc is a cone.
The top of the cone should be centered rotationally within the FOV.
The top of the cone is also positioned to match the bottom of the scroll lining interiors for the cases (i.e. this is the bottom position from which acquisition should proceed upward).
A cone was selected in place of a pin since it is easier to manufacture a cone that is placed at the correct height and is not brittle or easily bent.
"""

import tempfile
from build123d import *
import scrollcase as sc
from ocp_vscode import *
from meshlib import mrmeshpy as mm

NO_SCROLL = 0


def build_disc():
    case = sc.case.ScrollCase(scroll_height_mm=NO_SCROLL, scroll_radius_mm=NO_SCROLL)

    # Base
    with BuildPart(
        Location((0, 0, case.cylinder_bottom - case.square_height_mm))
    ) as mount_disc:
        cyl = Cylinder(
            case.mount_disc_diameter_mm / 2,
            case.mount_disc_height_mm,
            align=(Align.CENTER, Align.CENTER, Align.MAX),
        )

        # Cone
        Cone(
            bottom_radius=case.cylinder_top_to_lining_bottom,
            top_radius=0,
            height=case.cylinder_top_to_lining_bottom,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )

        # Alignment Cube
        with BuildSketch(cyl.faces().sort_by()[0]):
            Rectangle(case.mount_disc_box_width_mm, case.mount_disc_box_width_mm)
        extrude(amount=-case.mount_disc_box_height_mm, mode=Mode.SUBTRACT)

        # Bolt holes
        with Locations(
            (0, case.mount_disc_diameter_mm / 2, -case.mount_disc_height_mm / 2)
        ):
            Cylinder(
                case.mount_disc_hole_diameter_mm / 2,
                2 * case.mount_disc_hole_depth_mm,
                rotation=(90, 0, 0),
                mode=Mode.SUBTRACT,
            )
        with Locations(
            (0, -case.mount_disc_diameter_mm / 2, -case.mount_disc_height_mm / 2)
        ):
            Cylinder(
                case.mount_disc_hole_diameter_mm / 2,
                2 * case.mount_disc_hole_depth_mm,
                rotation=(-90, 0, 0),
                mode=Mode.SUBTRACT,
            )

    # Convert to mesh
    mount_disc_mesh = sc.mesh.brep_to_mesh(mount_disc.solids()[0])

    return mount_disc_mesh


disc = build_disc()

mm.saveMesh(disc, Path("disc.stl"))
