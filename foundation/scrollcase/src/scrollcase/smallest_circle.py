import numpy as np
import random
from typing import Tuple
import sys

sys.setrecursionlimit(1000000)


def _is_in_circle(pt: np.ndarray, center: np.ndarray, radius: float) -> bool:
    """Check if point `pt` is inside or on the circle defined by (center, radius).

    Args:
        pt (np.ndarray): The point to check.
        center (np.ndarray): The center of the circle.
        radius (float): The radius of the circle.

    Returns:
        bool: True if the point is inside or on the circle, False otherwise.
    """
    return bool(np.linalg.norm(pt - center) <= radius + 1e-14)


def _circle_two_points(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return the circle (center, radius) defined by two points p1 and p2.

    Args:
        p1 (np.ndarray): The first point.
        p2 (np.ndarray): The second point.

    Returns:
        Tuple[np.ndarray, float]: The center and radius of the circle.
    """
    center = (p1 + p2) / 2.0
    radius = np.linalg.norm(p1 - center)
    return center, float(radius)


def _circle_three_points(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Return the circle (center, radius) defined by three non-collinear points.

    If the points are collinear, this function may fail or produce a very large circle
    (handling collinearity outside of this function is recommended if needed).

    Args:
        p1 (np.ndarray): The first point.
        p2 (np.ndarray): The second point.
        p3 (np.ndarray): The third point.

    Returns:
        Tuple[np.ndarray, float]: The center and radius of the circle.
    """
    d = 2 * (
        p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])
    )

    if abs(d) < 1e-14:
        min_c, min_r = _circle_two_points(p1, p2)
        # Expand if needed to cover p3
        if not _is_in_circle(p3, min_c, min_r):
            c2, r2 = _circle_two_points(p1, p3)
            if r2 < min_r:
                min_c, min_r = c2, r2
            if not _is_in_circle(p2, c2, r2):
                c3, r3 = _circle_two_points(p2, p3)
                if r3 < min_r:
                    min_c, min_r = c3, r3
        return min_c, min_r

    ux = (
        np.sum(p1**2) * (p2[1] - p3[1])
        + np.sum(p2**2) * (p3[1] - p1[1])
        + np.sum(p3**2) * (p1[1] - p2[1])
    ) / d
    uy = (
        np.sum(p1**2) * (p3[0] - p2[0])
        + np.sum(p2**2) * (p1[0] - p3[0])
        + np.sum(p3**2) * (p2[0] - p1[0])
    ) / d

    center = np.array([ux, uy], dtype=float)
    radius = np.linalg.norm(p1 - center)
    return center, float(radius)


def _make_circle(boundary_points: list[np.ndarray]) -> Tuple[np.ndarray, float]:
    """Given up to 3 boundary points, return the circle (center, radius) passing through them.

    Args:
        boundary_points (list[np.ndarray]): List of boundary points.

    Returns:
        Tuple[np.ndarray, float]: The center and radius of the circle.
    """
    if not boundary_points:
        return np.array([0.0, 0.0]), 0.0
    elif len(boundary_points) == 1:
        # Only one boundary point => radius=0, center=the point itself
        return boundary_points[0], 0.0
    elif len(boundary_points) == 2:
        return _circle_two_points(boundary_points[0], boundary_points[1])
    else:
        return _circle_three_points(
            boundary_points[0], boundary_points[1], boundary_points[2]
        )


def _welzl(
    points: list[np.ndarray], boundary_points: list[np.ndarray], n: int
) -> Tuple[np.ndarray, float]:
    """Recursive function for Welzl's algorithm.

    Args:
        points (list[np.ndarray]): List of points (subset under consideration).
        boundary_points (list[np.ndarray]): Up to three points that define the current circle.
        n (int): Index up to which we are considering points (in `points`).

    Returns:
        Tuple[np.ndarray, float]: The center and radius of the circle.
    """
    if n == 0 or len(boundary_points) == 3:
        return _make_circle(boundary_points)

    p = points[n - 1]
    center, radius = _welzl(points, boundary_points, n - 1)

    if _is_in_circle(p, center, radius):
        return center, radius
    else:
        # Otherwise, p must belong to the boundary of the new circle
        return _welzl(points, boundary_points + [p], n - 1)


def smallest_enclosing_circle(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return the center (2D) and radius of the smallest enclosing circle
    of the input points using Welzl's algorithm.

    Args:
        points (np.ndarray): numpy array of shape (N, 2), each row a 2D point.

    Returns:
        Tuple[np.ndarray, float]: The center and radius of the smallest enclosing circle.
    """
    pts_list = [p for p in points]
    random.shuffle(pts_list)

    center, radius = _welzl(pts_list, [], len(pts_list))
    return center, radius
