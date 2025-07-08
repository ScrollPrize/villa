import logging
from build123d import *

logger = logging.getLogger(__name__)


def divider_curve(lining_outer_radius, wall_thickness_mm):
    pts = [
        (-lining_outer_radius + wall_thickness_mm, 0),
        (-lining_outer_radius / 2, lining_outer_radius / 5),
        (0, 0),
        (lining_outer_radius / 2, -lining_outer_radius / 5),
        (lining_outer_radius - wall_thickness_mm, 0),
    ]
    ln1 = ThreePointArc(pts[0], pts[1], pts[2])
    ln2 = ThreePointArc(pts[2], pts[3], pts[4])
    return ln1 + ln2


def divider_solid(lining_outer_radius, square_loft_radius, wall_thickness_mm):
    with BuildPart() as divider:
        with BuildSketch():
            with BuildLine() as divider_ln:
                ln1 = Line(
                    (-square_loft_radius * 2, 0),
                    (-lining_outer_radius + wall_thickness_mm, 0),
                )
                ln2 = divider_curve(lining_outer_radius, wall_thickness_mm)
                ln3 = Line(
                    (lining_outer_radius - wall_thickness_mm, 0),
                    (square_loft_radius * 2, 0),
                )

                ln4 = Line(
                    (square_loft_radius * 2, 0),
                    (square_loft_radius * 2, square_loft_radius * 2),
                )
                ln5 = Line(
                    (square_loft_radius * 2, square_loft_radius * 2),
                    (-square_loft_radius * 2, square_loft_radius * 2),
                )
                ln6 = Line(
                    (-square_loft_radius * 2, square_loft_radius * 2),
                    (-square_loft_radius * 2, 0),
                )
            make_face()
        extrude(amount=1000)

    return divider
