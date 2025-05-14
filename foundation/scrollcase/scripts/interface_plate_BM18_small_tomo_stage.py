from pathlib import Path
from math import sqrt

from bd_warehouse.thread import IsoThread
from build123d import *
import scrollcase as sc
from ocp_vscode import *
from meshlib import mrmeshpy as mm

NO_SCROLL = 0


def build_thread(case: sc.case.ScrollCase):
    with BuildPart() as thread:
        add(
            IsoThread(
                major_diameter=6,
                pitch=1,
                length=case.square_height_mm,
                external=False,
                end_finishes=("raw", "raw"),
                align=(Align.CENTER, Align.CENTER, Align.MAX),
            )
        )

        Box(
            case.square_height_mm * 2,
            case.square_height_mm * 2,
            case.square_height_mm * 2,
            mode=Mode.SUBTRACT,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )

        with Locations((0, 0, -case.square_height_mm)):
            Box(
                case.square_height_mm * 2,
                case.square_height_mm * 2,
                case.square_height_mm * 2,
                mode=Mode.SUBTRACT,
                align=(Align.CENTER, Align.CENTER, Align.MAX),
            )

    return thread


def build_interface_plate(thread_holes: bool = False):
    case = sc.case.ScrollCase(scroll_height_mm=NO_SCROLL, scroll_radius_mm=NO_SCROLL)

    with BuildPart() as interface_plate:
        add(
            sc.case.bottom_cap(
                case, with_bolt_protrusions=False, with_counter_bore=False
            )
        )

        # Bolt holes for M6s from scroll case above
        with Locations(
            (
                -case.base_bolt_hole_spacing_from_center_mm,
                -case.base_bolt_hole_spacing_from_center_mm,
                case.square_height_mm,
            ),
            (
                case.base_bolt_hole_spacing_from_center_mm,
                -case.base_bolt_hole_spacing_from_center_mm,
                case.square_height_mm,
            ),
            (
                -case.base_bolt_hole_spacing_from_center_mm,
                case.base_bolt_hole_spacing_from_center_mm,
                case.square_height_mm,
            ),
            (
                case.base_bolt_hole_spacing_from_center_mm,
                case.base_bolt_hole_spacing_from_center_mm,
                case.square_height_mm,
            ),
        ):
            if thread_holes:
                Cylinder(
                    3,
                    case.square_height_mm,
                    mode=Mode.SUBTRACT,
                    align=(Align.CENTER, Align.CENTER, Align.MAX),
                )
                add(build_thread(case))
            else:
                Cylinder(
                    case.base_bolt_hole_diameter_for_tapping_mm / 2,
                    case.square_height_mm,
                    mode=Mode.SUBTRACT,
                    align=(Align.CENTER, Align.CENTER, Align.MAX),
                )

        # Bolt holes for tomo stage below
        with Locations(
            (
                case.tomo_stage_bolt_hole_spacing_from_center_mm,
                case.tomo_stage_bolt_hole_spacing_from_center_mm,
                case.square_height_mm,
            ),
            (
                -case.tomo_stage_bolt_hole_spacing_from_center_mm,
                case.tomo_stage_bolt_hole_spacing_from_center_mm,
                case.square_height_mm,
            ),
            (
                case.tomo_stage_bolt_hole_spacing_from_center_mm,
                -case.tomo_stage_bolt_hole_spacing_from_center_mm,
                case.square_height_mm,
            ),
            (
                -case.tomo_stage_bolt_hole_spacing_from_center_mm,
                -case.tomo_stage_bolt_hole_spacing_from_center_mm,
                case.square_height_mm,
            ),
        ):
            Cylinder(
                case.base_bolt_hole_diameter_mm / 2,
                case.square_height_mm,
                mode=Mode.SUBTRACT,
                align=(Align.CENTER, Align.CENTER, Align.MAX),
            )
            Cylinder(
                case.base_bolt_hole_counter_bore_diameter_mm / 2,
                case.base_bolt_hole_counter_bore_depth_mm,
                mode=Mode.SUBTRACT,
                align=(Align.CENTER, Align.CENTER, Align.MAX),
            )

        # Kinematic mount pins
        with Locations((0, 0, case.square_height_mm)):
            with PolarLocations(
                case.kinematic_mount_slot_pos_radius_mm,
                case.kinematic_mount_num_slots,
                start_angle=90,
            ):
                Cone(
                    case.kinematic_mount_slot_width_mm * sqrt(2) / 2 + 1,
                    0,
                    case.kinematic_mount_slot_width_mm * sqrt(2) / 2 + 1,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                )

    show(interface_plate, reset_camera=Camera.KEEP)

    return interface_plate


interface_plate = build_interface_plate(thread_holes=True)

# Convert to mesh
interface_plate_mesh = sc.mesh.brep_to_mesh(interface_plate.solids()[0])

mm.saveMesh(interface_plate_mesh, Path("interface_plate.stl"))
