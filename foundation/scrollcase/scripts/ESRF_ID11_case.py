from pathlib import Path

from build123d import *
import scrollcase as sc
from ocp_vscode import show, Camera
from meshlib import mrmeshpy as mm

NO_SCROLL = 0


def build_case():
    config = sc.case.ScrollCaseConfig(
        scroll_height_mm=NO_SCROLL, scroll_radius_mm=NO_SCROLL
    )

    with BuildPart() as case:
        add(sc.case.ESRF_ID11_base(config))

    show(case, reset_camera=Camera.KEEP)

    return case


case = build_case()

# Convert to mesh
# case_mesh = sc.mesh.brep_to_mesh(case.solids()[0])

# mm.saveMesh(disc_mesh, Path("disc.stl"))
