from pathlib import Path

from build123d import *  # type: ignore
from meshlib import mrmeshpy as mm
from ocp_vscode import show, Camera

import scrollcase as sc


def build_case(config: sc.case.ScrollCaseConfig):
    with BuildPart() as case:
        add(sc.case.ESRF_ID11_base(config))

        wall, solid = sc.curved_divider_wall.divider_wall_and_solid(
            config.lining_outer_radius,
            config.wall_thickness_mm,
            config.id11_cylinder_height,
            case_max_radius=config.esrf_id11_diffractometer_plate_width_mm / 2,
        )
        add(wall)

        case_part = case.part
        divider_solid_part = solid.part
        assert isinstance(case_part, Part)
        assert isinstance(divider_solid_part, Part)

        divider_solid_part = divider_solid_part.move(
            Location((0, 0, config.square_height_mm))
        )

        left = case_part - divider_solid_part
        right = case_part & divider_solid_part

    return left, right


if __name__ == "__main__":
    EXAMPLE_SCROLL_DIM = 50

    config = sc.case.ScrollCaseConfig(
        scroll_height_mm=EXAMPLE_SCROLL_DIM,
        scroll_radius_mm=EXAMPLE_SCROLL_DIM / 2,
    )

    left, right = build_case(config)

    show(left, right, reset_camera=Camera.KEEP)

    # Convert to mesh
    # case_mesh = sc.mesh.brep_to_mesh(case.solids()[0])

    # mm.saveMesh(disc_mesh, Path("ESRF_ID11_case.stl"))
