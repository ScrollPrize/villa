import numpy as np
from scipy.spatial import ConvexHull
import logging
from . import smallest_circle
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

def _project_points_onto_plane(
    points: np.ndarray, normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project points onto a plane with given unit normal.
    
    Returns:
      points_global: Points projected onto the plane (in global coords).
      points_local: 2D coordinates in the plane.
      d:           Distances along the normal.
      basis:       The two basis vectors for the plane.
    """
    # Use a fixed reference vector (assumes normal is not parallel to [0,1,0])
    reference = np.array([0, 1, 0])
    if abs(np.dot(normal, reference)) > 0.99:
        reference = np.array([1, 0, 0])
    u = np.cross(normal, reference)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    d = np.dot(points, normal)
    points_global = points - d[:, None] * normal
    points_local = np.hstack([np.dot(points_global, u)[:, None], np.dot(points_global, v)[:, None]])
    return points_global, points_local, d, np.vstack([u, v])

def fit_cylinder(
    points: np.ndarray, tol: float = 1e-4, _debug: bool = False
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Robustly fit a cylinder to the input points.
    
    Returns:
        rotation_matrix: 3x3 rotation matrix transforming points into cylinder coords.
        translation_vector: Translation to align the cylinder so that its bottom is at Z=0.
        radius: Fitted cylinder radius.
        height: Height of the point cloud along the cylinder axis.
    """
    # Compute convex hull to reduce the number of points
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    # --- Step 1: PCA for an initial guess ---
    mean = np.mean(hull_points, axis=0)
    cov = np.cov(hull_points - mean, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    principal_axis = eigenvectors[:, idx[0]]  # Candidate long axis
    n0 = principal_axis  # use as initial guess

    # --- Step 2: Augmented objective function ---
    def objective(normal):
        normal_unit = normal / np.linalg.norm(normal)
        _, projection_local, d, _ = _project_points_onto_plane(hull_points, normal_unit)
        _, radius = smallest_circle.smallest_enclosing_circle(projection_local)
        return radius

    constraints = [{"type": "eq", "fun": lambda n: np.dot(n, n) - 1}]
    result = minimize(objective, n0, constraints=constraints, method="SLSQP", tol=tol)
    best_normal = result.x / np.linalg.norm(result.x)
    best_radius = result.fun

    logger.info(f"Robust fit: best direction: {best_normal} with objective value {best_radius}")

    # --- Step 3: Recompute circle and cylinder parameters ---
    projection_global, projection_local, d, basis = _project_points_onto_plane(hull_points, best_normal)
    center_2d, radius = smallest_circle.smallest_enclosing_circle(projection_local)
    height = max(d) - min(d)

    # Build the rotation matrix. The two basis vectors become X & Y, and best_normal is Z.
    rotation_matrix = np.column_stack([basis.T, best_normal])
    rotation_matrix = rotation_matrix.T  # so that applying it rotates points into cylinder coords

    # Compute global center of the circle in the plane and translation such that bottom is at Z=0.
    global_center = basis.T @ center_2d
    translation_vector = rotation_matrix @ (-global_center)
    translation_vector[2] -= min(d)

    transformed_coords = rotation_matrix @ (hull_points.T) + translation_vector[:, None]

    if _debug:
        transformed_coords = transformed_coords.T
        # transformed_coords[:, 2] -= z_translation

        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.linspace(0, max(d) - min(d), 100)
        Theta, Z = np.meshgrid(theta, z)
        X = radius * np.cos(Theta)
        Y = radius * np.sin(Theta)

        scatter_points = go.Scatter3d(
            x=transformed_coords[:, 0],
            y=transformed_coords[:, 1],
            z=transformed_coords[:, 2],
            mode="markers",
            marker=dict(opacity=0.5, size=2),
        )

        # Create the surface plot
        surface = go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="reds",
            opacity=0.1,
            showscale=False,
            contours=dict(z=dict(show=True, color="black", width=1)),
        )

        # Combine both plots
        fig = go.Figure(data=[scatter_points, surface])

        # Set the aspect ratio and layout
        fig.update_layout(
            scene={"camera_projection_type": "orthographic", "aspectmode": "data"},
            showlegend=False,
            width=1000,
            height=1000,
        )

        fig.show(renderer="browser")

    return rotation_matrix, translation_vector, radius, height
