from pathlib import Path
from math import sqrt

from bd_warehouse.thread import IsoThread
from build123d import *
import scrollcase as sc
from ocp_vscode import *
from meshlib import mrmeshpy as mm

NO_SCROLL = 0
RISER_HEIGHT = 150


def build_riser():
    case = sc.case.ScrollCaseConfig(
        scroll_height_mm=NO_SCROLL, scroll_radius_mm=NO_SCROLL
    )

    with BuildPart() as riser:
        # Wide base cylinder
        Cylinder(
            radius=58,
            height=case.square_height_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )

        # Bolt holes for tomo stage below
        with Locations(
            (
                0,
                2 * case.tomo_stage_bolt_hole_spacing_from_center_mm,
                case.square_height_mm,
            ),
            (
                2 * case.tomo_stage_bolt_hole_spacing_from_center_mm,
                0,
                case.square_height_mm,
            ),
            (
                0,
                -2 * case.tomo_stage_bolt_hole_spacing_from_center_mm,
                case.square_height_mm,
            ),
            (
                -2 * case.tomo_stage_bolt_hole_spacing_from_center_mm,
                0,
                case.square_height_mm,
            ),
        ):
            Cylinder(
                case.m6_clearance_hole_diameter_semi_loose_mm / 2,
                case.square_height_mm,
                mode=Mode.SUBTRACT,
                align=(Align.CENTER, Align.CENTER, Align.MAX),
            )
            Cylinder(
                case.m6_head_counter_bore_diameter_mm / 2,
                case.m6_head_counter_bore_depth_mm,
                mode=Mode.SUBTRACT,
                align=(Align.CENTER, Align.CENTER, Align.MAX),
            )

        # Tall cylinder
        with BuildPart() as tall_cylinder:
            with Locations(
                (
                    0,
                    0,
                    case.square_height_mm,
                ),
            ):
                Cylinder(
                    radius=50,
                    height=RISER_HEIGHT - case.square_height_mm,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                )

            # Cutouts for access to bottom bolts
            with PolarLocations(
                radius=2 * case.tomo_stage_bolt_hole_spacing_from_center_mm,
                count=4,
                start_angle=0,
            ):
                Cylinder(
                    radius=10,
                    height=100,
                    mode=Mode.SUBTRACT,
                    rotation=(0, 15, 0),
                )

        # Tappable holes in top for bolts from interface plate above
        with Locations(
            (
                case.tomo_stage_bolt_hole_spacing_from_center_mm,
                case.tomo_stage_bolt_hole_spacing_from_center_mm,
                RISER_HEIGHT,
            ),
            (
                -case.tomo_stage_bolt_hole_spacing_from_center_mm,
                case.tomo_stage_bolt_hole_spacing_from_center_mm,
                RISER_HEIGHT,
            ),
            (
                case.tomo_stage_bolt_hole_spacing_from_center_mm,
                -case.tomo_stage_bolt_hole_spacing_from_center_mm,
                RISER_HEIGHT,
            ),
            (
                -case.tomo_stage_bolt_hole_spacing_from_center_mm,
                -case.tomo_stage_bolt_hole_spacing_from_center_mm,
                RISER_HEIGHT,
            ),
        ):
            Cylinder(
                case.m6_clearance_hole_diameter_for_tapping_mm / 2,
                case.square_height_mm * 3,
                mode=Mode.SUBTRACT,
                align=(Align.CENTER, Align.CENTER, Align.MAX),
            )

    show(riser, reset_camera=Camera.KEEP)

    return riser


riser = build_riser()

# Convert to mesh
riser_mesh = sc.mesh.brep_to_mesh(riser.solids()[0])

mm.saveMesh(riser_mesh, Path("riser.stl"))
