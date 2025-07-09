from argparse import ArgumentParser
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

        with Locations(
            (0, 0, config.square_height_mm + 5), (0, 0, config.id11_cylinder_height - 5)
        ):
            Cylinder(
                config.m4_clearance_hole_diameter_tight_mm / 2,
                10,
                align=(Align.CENTER, Align.CENTER, Align.CENTER),
                mode=Mode.SUBTRACT,
                rotation=(90, -42, 0),
            )

        case_part = case.part
        divider_solid_part = solid.part
        assert isinstance(case_part, Part)
        assert isinstance(divider_solid_part, Part)

        divider_solid_part = divider_solid_part.move(
            Location((0, 0, config.square_height_mm))
        )

        left = (case_part - divider_solid_part).solid()
        right = (case_part & divider_solid_part).solid()

        assert isinstance(left, Solid)
        assert isinstance(right, Solid)

    return left, right


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--output-dir", default="test_output")
    parser.add_argument("--scroll-name", default="TEST_SCROLL")
    args = parser.parse_args()

    if args.mesh is None:
        EXAMPLE_SCROLL_DIM = 50

        config = sc.case.ScrollCaseConfig(
            scroll_height_mm=EXAMPLE_SCROLL_DIM,
            scroll_radius_mm=EXAMPLE_SCROLL_DIM / 2,
        )

        case_left, case_right = build_case(config)

        show(case_left, case_right, reset_camera=Camera.KEEP)
    else:
        assert args.mesh.is_file()

        # Load it first just to get the height
        scroll_mesh = sc.mesh.ScrollMesh(args.mesh)
        (
            _,
            _,
            _,
            _,
            _,
            _,
            height,
        ) = sc.mesh.build_lining(scroll_mesh)

        # Fixed 60mm vertical target from ID11 stage specs
        vertical_offset = 60 - height / 2

        scroll_mesh = sc.mesh.ScrollMesh(args.mesh, vertical_offset=vertical_offset)
        (
            lining_mesh_pos,
            lining_mesh_neg,
            cavity_mesh_pos,
            cavity_mesh_neg,
            mesh_scroll,
            radius,
            height,
        ) = sc.mesh.build_lining(scroll_mesh)

        config = sc.case.ScrollCaseConfig(
            height, radius, label_line_1=f"PHerc{args.scroll_name}", label_line_2="v1"
        )
        case_left, case_right = build_case(config)

        # Combine the BRep case halves with the mesh lining.
        combined_mesh_right = sc.mesh.combine_brep_case_lining(
            case_right, cavity_mesh_pos, lining_mesh_pos
        )
        combined_mesh_left = sc.mesh.combine_brep_case_lining(
            case_left, cavity_mesh_neg, lining_mesh_neg
        )

        scroll_stl_path = Path(args.output_dir) / f"{args.scroll_name}_scroll.stl"
        right_stl_path = Path(args.output_dir) / f"{args.scroll_name}_right.stl"
        left_stl_path = Path(args.output_dir) / f"{args.scroll_name}_left.stl"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        mm.saveMesh(mesh_scroll, scroll_stl_path)
        mm.saveMesh(combined_mesh_right, right_stl_path)
        mm.saveMesh(combined_mesh_left, left_stl_path)
