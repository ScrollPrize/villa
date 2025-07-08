from pathlib import Path

from build123d import *  # type: ignore
from meshlib import mrmeshpy as mm
from ocp_vscode import show, Camera

import scrollcase as sc

NO_SCROLL = 0


def build_case():
    config = sc.case.ScrollCaseConfig(
        scroll_height_mm=NO_SCROLL, scroll_radius_mm=NO_SCROLL
    )

    with BuildPart() as case:
        add(sc.case.ESRF_ID11_base(config))

    show(case, reset_camera=Camera.KEEP)

    return left, right


left, right = build_case()

# Convert to mesh
# case_mesh = sc.mesh.brep_to_mesh(case.solids()[0])

# mm.saveMesh(disc_mesh, Path("ESRF_ID11_case.stl"))
