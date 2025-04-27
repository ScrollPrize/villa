import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from build123d import *

from . import divider_utils

logger = logging.getLogger(__name__)


@dataclass
class ScrollCase:
    # Z=0 is the scroll bottom

    # Scroll dimensions defined per-scroll
    scroll_height_mm: float
    scroll_radius_mm: float

    # Labels defined per-scroll (remaining params fixed across scrolls)
    label_line_1: str = ""
    label_line_2: str = ""

    # Gap/offset between scroll and lining wall interior
    lining_offset_mm: float = 2
    # Wall thicknesses
    wall_thickness_mm: float = 2

    # Between lining exterior and top/bottom caps
    upper_margin_mm: float = 3
    lower_margin_mm: float = 3

    # Square caps
    square_height_mm: float = 10
    square_edge_fillet: float = 6.25
    right_cap_buffer: float = 0.5
    # Based on M4 bolts
    cap_bolt_hole_diameter_mm: float = 5
    cap_bolt_counter_bore_diameter_mm: float = 8
    cap_bolt_counter_bore_depth_mm: float = 2
    cap_bolt_nut_diameter_mm: float = 9
    cap_bolt_nut_depth_mm: float = 3.5

    # Mounting disc
    mount_disc_diameter_mm: float = 112.5
    mount_disc_height_mm: float = 10
    kinematic_mount_num_slots: int = 3
    kinematic_mount_slot_pos_radius_mm: float = (mount_disc_diameter_mm / 2) * (2 / 3)
    kinematic_mount_slot_width_mm: float = 2
    kinematic_mount_slot_length_mm: float = 10

    # Text properties
    text_font_size: float = 8
    text_depth_mm: float = 0.5

    # Base bolt holes (based on M6)
    base_bolt_hole_diameter_mm: float = 6.8
    base_bolt_hole_counter_bore_diameter_mm: float = 10.5
    base_bolt_hole_counter_bore_depth_mm: float = 5
    base_bolt_hole_spacing_from_center_mm: float = 50

    @property
    def lining_outer_radius(self):
        return self.scroll_radius_mm + self.lining_offset_mm + self.wall_thickness_mm

    @property
    def lining_outer_diameter(self):
        return 2 * self.lining_outer_radius

    @property
    def cylinder_height(self):
        return (
            self.scroll_height_mm
            + 2 * self.lining_offset_mm
            + 2 * self.wall_thickness_mm
            + self.lower_margin_mm
            + self.upper_margin_mm
        )

    @property
    def cylinder_bottom(self):
        return -self.lining_offset_mm - self.wall_thickness_mm - self.lower_margin_mm

    @property
    def square_loft_radius(self):
        return max(self.mount_disc_diameter_mm / 2, self.lining_outer_radius)

    @property
    def lining_interior_height(self):
        return 2 * self.lining_offset_mm + self.scroll_height_mm

    @property
    def mount_disc_top_to_lining_bottom(self):
        return self.square_height_mm + self.lower_margin_mm + self.wall_thickness_mm


def hex_nut(diameter_mm: float, depth_mm: float):
    with BuildPart() as hex_part:
        with BuildSketch() as hex_sketch:
            RegularPolygon(radius=diameter_mm / 2, side_count=6)
        extrude(amount=depth_mm)

    return hex_part


def mount_disc(case: ScrollCase):
    with BuildPart() as mount_disc_part:
        cyl = Cylinder(
            case.mount_disc_diameter_mm / 2,
            case.mount_disc_height_mm,
            align=(Align.CENTER, Align.CENTER, Align.MAX),
        )

        # Kinematic mount slots
        with Locations((0, 0, -case.mount_disc_height_mm)):
            with PolarLocations(
                case.kinematic_mount_slot_pos_radius_mm,
                case.kinematic_mount_num_slots,
                start_angle=90,
            ):
                Box(
                    case.kinematic_mount_slot_length_mm,
                    case.kinematic_mount_slot_width_mm,
                    case.kinematic_mount_slot_width_mm,
                    rotation=(45, 0, 0),
                    mode=Mode.SUBTRACT,
                )

        return mount_disc_part


def cap(case: ScrollCase):
    with BuildPart() as cap_part:
        with BuildSketch():
            # Main rectangle
            r = Rectangle(2 * case.square_loft_radius, 2 * case.square_loft_radius)
            fillet(r.vertices(), case.square_edge_fillet)

            # Left bolt protrusion
            with Locations((-case.square_loft_radius, 0)):
                r2 = Rectangle(
                    case.square_height_mm,
                    case.square_height_mm * 2,
                    align=(Align.MAX, Align.CENTER),
                )
                fillet(r2.vertices(), case.square_height_mm / 2)

            # Right bolt protrusion
            with Locations((case.square_loft_radius, 0)):
                r3 = Rectangle(
                    case.square_height_mm,
                    case.square_height_mm * 2,
                    align=(Align.MIN, Align.CENTER),
                )
                fillet(r3.vertices(), case.square_height_mm / 2)
        extrude(amount=case.square_height_mm)

        # Bolt holes
        with Locations(
            (
                -case.square_loft_radius - case.square_height_mm / 2,
                0,
                case.square_height_mm / 2,
            ),
            (
                case.square_loft_radius + case.square_height_mm / 2,
                0,
                case.square_height_mm / 2,
            ),
        ):
            Cylinder(
                case.cap_bolt_hole_diameter_mm / 2,
                4 * case.square_height_mm,
                rotation=(90, 0, 0),
                mode=Mode.SUBTRACT,
            )

        # Bolt head cutouts
        with Locations(
            (
                -case.square_loft_radius - case.square_height_mm / 2,
                case.square_height_mm - case.cap_bolt_counter_bore_depth_mm,
                case.square_height_mm / 2,
            ),
            (
                case.square_loft_radius + case.square_height_mm / 2,
                case.square_height_mm - case.cap_bolt_counter_bore_depth_mm,
                case.square_height_mm / 2,
            ),
        ):
            Cylinder(
                case.cap_bolt_counter_bore_diameter_mm / 2,
                2 * case.square_height_mm,
                mode=Mode.SUBTRACT,
                rotation=(90, 0, 0),
                align=(Align.CENTER, Align.CENTER, Align.MAX),
            )

        # Hexagonal nut cutouts
        with Locations(
            (
                -case.square_loft_radius - case.square_height_mm / 2,
                -case.square_height_mm + case.cap_bolt_nut_depth_mm,
                case.square_height_mm / 2,
            ),
            (
                case.square_loft_radius + case.square_height_mm / 2,
                -case.square_height_mm + case.cap_bolt_nut_depth_mm,
                case.square_height_mm / 2,
            ),
        ):
            hex = hex_nut(
                case.cap_bolt_nut_diameter_mm, case.cap_bolt_nut_depth_mm + 10
            )
            add(hex, mode=Mode.SUBTRACT, rotation=(90, 0, 0))

    return cap_part


def top_cap(case: ScrollCase):
    with BuildPart() as top_cap_part:
        add(cap(case))

        # Text
        top_face = top_cap_part.faces().sort_by(Axis.Z)[-1]
        with BuildSketch(top_face):
            with Locations((0, 40)):
                Text(case.label_line_1, case.text_font_size)
            with Locations((0, 40 - case.square_loft_radius)):
                Text(case.label_line_1, case.text_font_size)
            with Locations((0, 30 - case.square_loft_radius)):
                Text(case.label_line_2, case.text_font_size)
            with Locations((0, 20 - case.square_loft_radius)):
                Text(
                    f"{case.lining_outer_diameter:.2f}D x {case.lining_interior_height:.2f}H",
                    case.text_font_size,
                )
        extrude(amount=-case.text_depth_mm, mode=Mode.SUBTRACT)

    return top_cap_part


def bottom_cap(case: ScrollCase):
    with BuildPart() as bottom_cap_part:
        add(cap(case))

        # Bolt holes
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

            # Alignment arrow
            with BuildSketch(Location((0, 0, case.square_height_mm))):
                Arrow(
                    case.square_height_mm / 2,
                    Line(
                        (0, case.square_loft_radius),
                        (0, case.square_loft_radius - case.square_height_mm),
                    ),
                    case.square_height_mm / 8,
                )
            extrude(amount=-case.text_depth_mm, mode=Mode.SUBTRACT)

    return bottom_cap_part


def build_case(case: ScrollCase) -> tuple[Solid, Solid]:
    """Build the scroll case.

    Args:
        case (ScrollCase): The scroll case parameters.

    Returns:
        tuple[Solid, Solid]: The left and right halves of the scroll case.
    """
    logger.info(
        f"Constructing case with scroll radius: {case.scroll_radius_mm}, height: {case.scroll_height_mm}"
    )

    with BuildPart(Location((0, 0, case.cylinder_bottom))) as case_part:
        # Top and bottom caps
        with Locations((0, 0, case.cylinder_height)):
            add(top_cap(case))
        with Locations((0, 0, -case.square_height_mm)):
            add(bottom_cap(case))

        with BuildPart() as divider_wall:
            with BuildLine() as spline_ln:
                ln = divider_utils.divider_curve(
                    case.lining_outer_radius,
                    case.wall_thickness_mm,
                )
                tangent = ln % 0.5
                orthogonal_plane = Plane(
                    origin=(0, 0, case.cylinder_bottom),
                    z_dir=tangent,
                )
            with BuildSketch(orthogonal_plane) as spline_sk:
                Rectangle(
                    case.wall_thickness_mm * 2,
                    case.cylinder_height,
                    align=(Align.CENTER, Align.MAX),
                )
            sweep()

            with BuildSketch(
                Plane(
                    origin=(0, 0, case.cylinder_bottom),
                    z_dir=(0, 0, 1),
                )
            ):
                with Locations(
                    (-case.lining_outer_radius + case.wall_thickness_mm, 0),
                    (case.lining_outer_radius - case.wall_thickness_mm, 0),
                ):
                    Circle(case.wall_thickness_mm)
            extrude(amount=case.cylinder_height)

        divider_solid_part = divider_utils.divider_solid(
            case.lining_outer_radius,
            case.square_loft_radius,
            case.wall_thickness_mm,
        ).part.move(Location((0, 0, case.cylinder_bottom - case.square_height_mm)))

        left = case_part.part - divider_solid_part
        right = case_part.part & divider_solid_part

    # Base
    with BuildPart(
        Location((0, 0, case.cylinder_bottom - case.square_height_mm))
    ) as base_disc:
        add(mount_disc(case))

        # Extra space at bottom of right case half
        with Locations((0, 0, -case.right_cap_buffer)):
            remove_part = divider_utils.divider_solid(
                case.lining_outer_radius,
                case.square_loft_radius,
                case.wall_thickness_mm,
            ).part
            add(remove_part, mode=Mode.SUBTRACT)

    left = left.solid() + base_disc.solid()
    right = right.solid()

    return left, right
