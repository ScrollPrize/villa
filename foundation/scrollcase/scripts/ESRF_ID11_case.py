from pathlib import Path

from build123d import *  # type: ignore
from meshlib import mrmeshpy as mm
from ocp_vscode import show, Camera

import scrollcase as sc

EXAMPLE_SCROLL_DIM = 50


def build_case(config: sc.case.ScrollCaseConfig):
    with BuildPart() as case:
        add(sc.case.ESRF_ID11_base(config))
        # add(sc.case.)

    show(case, reset_camera=Camera.KEEP)

    return case, case


if __name__ == "__main__":
    config = sc.case.ScrollCaseConfig(
        scroll_height_mm=EXAMPLE_SCROLL_DIM, scroll_radius_mm=EXAMPLE_SCROLL_DIM
    )

    left, right = build_case(config)

    # Convert to mesh
    # case_mesh = sc.mesh.brep_to_mesh(case.solids()[0])

    # mm.saveMesh(disc_mesh, Path("ESRF_ID11_case.stl"))
