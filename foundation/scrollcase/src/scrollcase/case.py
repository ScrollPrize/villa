import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from build123d import *

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
    lining_thickness_mm: float = 2
    wall_thickness_mm: float = 2

    # Between lining exterior and top/bottom caps
    upper_margin_mm: float = 3
    lower_margin_mm: float = 3

    # Between lining exterior and cylinder wall interior
    # Just enough space to not touch/create trap points for glass beads
    # Otherwise want to minimize exterior diameter
    radial_margin_mm: float = 0.5

    # Mounting disc dimensions
    mount_disc_diameter_mm: float = 112.5
    mount_disc_height_mm: float = 10
    mount_disc_hole_depth_mm: float = 5.75
    mount_disc_hole_diameter_mm: float = 6.8
    mount_disc_box_height_mm: float = 7
    mount_disc_box_width_mm: float = 13.5

    # Alignment ring spacing (up from bottom of lining interior)
    alignment_ring_spacing_mm: float = (
        98  # TODO(srparsons) this would be more helpful at I12 if it also accounts for the FOV
    )
    alignment_ring_width_mm: float = 1.5

    # Alignment nubs
    nub_size_mm: float = 3
    nub_depth_mm: float = 2
    nub_margin_mm: float = 1

    # Square caps
    square_height_mm: float = 10
    square_edge_fillet: float = 20
    oring_width: float = 3
    oring_depth: float = 2
    right_cap_buffer: float = 1

    # Text properties
    text_font_size: float = 8
    text_depth_mm: float = 0.5

    @property
    def lining_outer_radius(self):
        return self.scroll_radius_mm + self.lining_offset_mm + self.lining_thickness_mm

    @property
    def cylinder_height(self):
        return (
            self.scroll_height_mm
            + 2 * self.lining_offset_mm
            + 2 * self.lining_thickness_mm
            + self.lower_margin_mm
            + self.upper_margin_mm
        )

    @property
    def cylinder_inner_radius(self):
        return self.lining_outer_radius + self.radial_margin_mm

    @property
    def cylinder_outer_radius(self):
        return self.cylinder_inner_radius + self.wall_thickness_mm

    @property
    def cylinder_bottom(self):
        return -self.lining_offset_mm - self.lining_thickness_mm - self.lower_margin_mm

    @property
    def square_loft_radius(self):
        return max(self.mount_disc_diameter_mm / 2, self.cylinder_outer_radius)

    @property
    def cylinder_outer_diameter(self):
        return 2 * self.cylinder_outer_radius

    @property
    def lining_interior_height(self):
        return 2 * self.lining_offset_mm + self.scroll_height_mm

    @property
    def cylinder_top_to_lining_bottom(self):
        return self.square_height_mm + self.lower_margin_mm + self.lining_thickness_mm


def alignment_ring(case: ScrollCase):
    """Generate an alignment ring.

    Args:
        radius: Minor radius

    Returns:
        Alignment ring
    """
    with BuildPart() as part:
        Cylinder(
            case.cylinder_inner_radius + case.wall_thickness_mm,
            case.alignment_ring_width_mm,
        )
        Cylinder(
            case.cylinder_inner_radius, case.alignment_ring_width_mm, mode=Mode.SUBTRACT
        )

    return part


def honeycomb_cylinder(
    r: float, h: float, t: float, nr: float = 6, gap: float = 2
) -> Compound:
    """Build a hollow cylinder with honeycomb cutouts.

    Args:
        r: Cylinder minor radius.
        h: Cylinder height.
        t: Cylinder thickness.
        nr: Number of radial cutouts.
        gap: Gap between cutouts.

    Returns:
        Honeycomb cylinder.
    """
    # NOTE(akoen): This is not quite right since this assumes that the plane is flat,
    # which it is not.
    width = (4 * np.pi * r) / (3 * nr) - 2 * gap / np.sqrt(3)
    spacing = 3 / 2 * width + np.sqrt(3) * gap

    height = width * np.cos(np.pi / 6)

    nz = int(np.ceil(2 * h / (height + gap))) + 1

    all_hexes = []
    for iz in range(nz):
        start_angle = 360 / (2 * nr) * (iz % 2)
        hex = (
            (Plane.YZ * RegularPolygon(width / 2, 6, major_radius=True))
            .face()
            .translate((0, 0, iz * (height / 2 + gap / 2)))
        )
        hex = extrude(hex, 3, both=True)
        hexes = PolarLocations(r, int(nr), start_angle=start_angle) * hex
        all_hexes.extend(hexes)

    all_hexes = list(all_hexes)

    cyl = Cylinder(r + t, h, align=(Align.CENTER, Align.CENTER, Align.MIN)) - Cylinder(
        r, h, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )

    honey_cyl = cyl - all_hexes

    return honey_cyl


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

    honeycomb = honeycomb_cylinder(
        case.cylinder_inner_radius,
        case.cylinder_height,
        case.wall_thickness_mm,
        gap=3.0,
    )

    with BuildPart(Location((0, 0, case.cylinder_bottom))) as case_part:
        add(honeycomb)

        # Parting rect
        Box(
            case.cylinder_outer_diameter,
            2 * case.wall_thickness_mm,
            case.cylinder_height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )

        # Alignment rings
        if case.alignment_ring_spacing_mm:
            with Locations(
                [
                    (0.0, 0.0, x)
                    for x in np.arange(
                        case.lower_margin_mm + case.lining_thickness_mm,
                        case.cylinder_height,
                        case.alignment_ring_spacing_mm,
                    )
                ]  # type: ignore
            ):
                add(alignment_ring(case))

        # Top Cap
        with BuildSketch(Location((0, 0, case.cylinder_bottom + case.cylinder_height))):
            r = Rectangle(2 * case.square_loft_radius, 2 * case.square_loft_radius)
            fillet(r.vertices(), case.square_edge_fillet)

        cap = extrude(amount=case.square_height_mm)

        # Text
        with BuildSketch(cap.faces().sort_by()[-1]):
            with Locations((0, 40)):
                Text(case.label_line_1, case.text_font_size)
            with Locations((0, 40 - case.square_loft_radius)):
                Text(case.label_line_1, case.text_font_size)
            with Locations((0, 30 - case.square_loft_radius)):
                Text(case.label_line_2, case.text_font_size)
            with Locations((0, 20 - case.square_loft_radius)):
                Text(
                    f"{case.cylinder_outer_diameter:.2f}D x {case.lining_interior_height:.2f}H",
                    case.text_font_size,
                )
        extrude(amount=-case.text_depth_mm, mode=Mode.SUBTRACT)

        # Top O-ring
        with BuildSketch(
            Location(
                (
                    0,
                    0,
                    case.cylinder_bottom
                    + case.cylinder_height
                    + case.square_height_mm / 2,
                )
            )
        ):
            Rectangle(
                2 * case.square_loft_radius,
                2 * case.square_loft_radius,
            )
            with BuildSketch(mode=Mode.SUBTRACT):
                r = Rectangle(
                    2 * case.square_loft_radius - 2 * case.oring_depth,
                    2 * case.square_loft_radius - 2 * case.oring_depth,
                )
                fillet(r.vertices(), case.square_edge_fillet)

        extrude(amount=case.oring_width / 2, both=True, mode=Mode.SUBTRACT)

        # Bottom Cap
        with BuildSketch(Location((0, 0, case.cylinder_bottom))):
            r = Rectangle(2 * case.square_loft_radius, 2 * case.square_loft_radius)
            fillet(r.vertices(), case.square_edge_fillet)

        extrude(amount=-case.square_height_mm)

        # Bottom O-ring
        with BuildSketch(
            Location(
                (
                    0,
                    0,
                    case.cylinder_bottom - case.square_height_mm / 2,
                )
            )
        ):
            Rectangle(
                2 * case.square_loft_radius,
                2 * case.square_loft_radius,
            )
            with BuildSketch(mode=Mode.SUBTRACT):
                r = Rectangle(
                    2 * case.square_loft_radius - 2 * case.oring_depth,
                    2 * case.square_loft_radius - 2 * case.oring_depth,
                )
                fillet(r.vertices(), case.square_edge_fillet)

        extrude(amount=-case.oring_width / 2, both=True, mode=Mode.SUBTRACT)

        split(bisect_by=Plane.XZ, keep=Keep.BOTH)

    # Base
    with BuildPart(
        Location((0, 0, case.cylinder_bottom - case.square_height_mm))
    ) as mount_disc:
        cyl = Cylinder(
            case.mount_disc_diameter_mm / 2,
            case.mount_disc_height_mm,
            align=(Align.CENTER, Align.CENTER, Align.MAX),
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

    # Alignment nubs
    with BuildPart(mode=Mode.PRIVATE) as nubs:
        with Locations((0, 0, case.cylinder_bottom - case.square_height_mm / 2)):
            with Locations(((case.cylinder_inner_radius / 2), 0, 0)):
                Box(
                    case.nub_size_mm,
                    case.nub_depth_mm,
                    case.nub_size_mm,
                    align=(Align.CENTER, Align.MIN, Align.CENTER),
                )
            with Locations((-(case.cylinder_inner_radius / 2), 0, 0)):
                Box(
                    case.nub_size_mm,
                    case.nub_depth_mm,
                    case.nub_size_mm,
                    rotation=(0, 45, 0),
                    align=(Align.CENTER, Align.MIN, Align.CENTER),
                )

        with Locations(
            (
                0,
                0,
                case.cylinder_bottom + case.cylinder_height + case.square_height_mm / 2,
            )
        ):
            with Locations(((case.cylinder_inner_radius / 2), 0, 0)):
                Box(
                    case.nub_size_mm,
                    case.nub_depth_mm,
                    case.nub_size_mm,
                    align=(Align.CENTER, Align.MIN, Align.CENTER),
                )
            with Locations((-(case.cylinder_inner_radius / 2), 0, 0)):
                Box(
                    case.nub_size_mm,
                    case.nub_depth_mm,
                    case.nub_size_mm,
                    rotation=(0, 45, 0),
                    align=(Align.CENTER, Align.MIN, Align.CENTER),
                )

    # Alignment nub hollows
    with BuildPart() as hollows:
        with Locations((0, 0, case.cylinder_bottom - case.square_height_mm / 2)):
            with Locations(((case.cylinder_inner_radius / 2), 0, 0)):
                Box(
                    case.nub_size_mm + 2 * case.nub_margin_mm,
                    case.nub_depth_mm,
                    case.nub_size_mm + 2 * case.nub_margin_mm,
                    align=(Align.CENTER, Align.MIN, Align.CENTER),
                )
            with Locations((-(case.cylinder_inner_radius / 2), 0, 0)):
                Box(
                    case.nub_size_mm + 2 * case.nub_margin_mm,
                    case.nub_depth_mm,
                    case.nub_size_mm + 2 * case.nub_margin_mm,
                    rotation=(0, 45, 0),
                    align=(Align.CENTER, Align.MIN, Align.CENTER),
                )

        with Locations(
            (
                0,
                0,
                case.cylinder_bottom + case.cylinder_height + case.square_height_mm / 2,
            )
        ):
            with Locations(((case.cylinder_inner_radius / 2), 0, 0)):
                Box(
                    case.nub_size_mm + 2 * case.nub_margin_mm,
                    case.nub_depth_mm,
                    case.nub_size_mm + 2 * case.nub_margin_mm,
                    align=(Align.CENTER, Align.MIN, Align.CENTER),
                )
            with Locations((-(case.cylinder_inner_radius / 2), 0, 0)):
                Box(
                    case.nub_size_mm + 2 * case.nub_margin_mm,
                    case.nub_depth_mm,
                    case.nub_size_mm + 2 * case.nub_margin_mm,
                    rotation=(0, 45, 0),
                    align=(Align.CENTER, Align.MIN, Align.CENTER),
                )

    # Extra space at bottom of right case half
    with BuildPart() as right_bottom_margin:
        with Locations((0, 0, case.cylinder_bottom - case.square_height_mm)):
            Box(
                case.square_loft_radius * 2,
                case.square_loft_radius * 2,
                case.right_cap_buffer,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
            )

    left = case_part.solids()[0] + mount_disc.solids()[0] + nubs.solids()
    right = case_part.solids()[1] - hollows.solids() - right_bottom_margin.solids()

    return left, right
