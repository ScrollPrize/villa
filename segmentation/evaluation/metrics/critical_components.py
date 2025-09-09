"""
Module: critical_components
Computes the numbers of positive and negative critical components for 3D images.
"""

from typing import Tuple, Set, Dict, Any, List
import numpy as np
from random import sample
from scipy.ndimage import label

def false_negative_mask(y_target: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes the false negative mask.

    Parameters
    ----------
    y_target : np.ndarray
        Groundtruth segmentation with unique labels.
    y_pred : np.ndarray
        Predicted segmentation with unique labels.

    Returns
    -------
    np.ndarray
        Binary mask with false negatives marked as 1.
    """
    false_negatives: np.ndarray = y_target.astype(bool) * (1 - y_pred.astype(bool))
    return false_negatives.astype(int)

def get_foreground(img: np.ndarray) -> Set[Tuple[int, int, int]]:
    """
    Retrieves the set of foreground voxel coordinates from a 3D image.

    Parameters
    ----------
    img : np.ndarray
        Input 3D image.

    Returns
    -------
    Set[Tuple[int, int, int]]
        Set of voxel coordinates in the foreground.
    """
    x, y, z = np.nonzero(img)
    return {(int(x[i]), int(y[i]), int(z[i])) for i in range(len(x))}

def get_nbs(xyz: Tuple[int, int, int], shape: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """
    Retrieves the 26-connected neighbors of a voxel.

    Parameters
    ----------
    xyz : Tuple[int, int, int]
        Coordinates of the voxel.
    shape : Tuple[int, int, int]
        Shape of the 3D image.

    Returns
    -------
    List[Tuple[int, int, int]]
        List of neighbor voxel coordinates.
    """
    x_offsets, y_offsets, z_offsets = np.meshgrid(
        [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"
    )
    nbs = np.column_stack(
        [
            (xyz[0] + x_offsets).ravel(),
            (xyz[1] + y_offsets).ravel(),
            (xyz[2] + z_offsets).ravel(),
        ]
    )
    mask = np.all((nbs >= 0) & (nbs < np.array(shape)), axis=1)
    return [tuple(coord) for coord in nbs[mask]]

def extract_component(
    y_target: np.ndarray,
    y_mistakes: np.ndarray,
    y_minus_mistakes: np.ndarray,
    xyz_r: Tuple[int, int, int]
) -> Tuple[np.ndarray, Set[Tuple[int, int, int]], bool]:
    """
    Extracts a connected component via BFS starting from a given root voxel.

    Parameters
    ----------
    y_target : np.ndarray
        Groundtruth segmentation.
    y_mistakes : np.ndarray
        Binary mask indicating mistakes.
    y_minus_mistakes : np.ndarray
        Connected components of y_target without mistakes.
    xyz_r : Tuple[int, int, int]
        Root voxel coordinate.

    Returns
    -------
    Tuple[np.ndarray, Set[Tuple[int, int, int]], bool]
        A binary mask of the component, the set of visited voxels, and a flag
        indicating if the component is critical.
    """
    mask = np.zeros(y_target.shape, dtype=bool)
    collisions: Dict[int, int] = {}
    is_critical: bool = False
    queue: List[Tuple[int, int, int]] = [xyz_r]
    visited: Set[Tuple[int, int, int]] = set()

    while queue:
        xyz_i = queue.pop(0)
        mask[xyz_i] = True
        for xyz_j in get_nbs(xyz_i, y_target.shape):
            if xyz_j not in visited and y_target[xyz_r] == y_target[xyz_j]:
                visited.add(xyz_j)
                if y_mistakes[xyz_j] == 1:
                    queue.append(xyz_j)
                elif not is_critical:
                    key = int(y_target[xyz_j])
                    if key not in collisions:
                        collisions[key] = int(y_minus_mistakes[xyz_j])
                    elif collisions[key] != int(y_minus_mistakes[xyz_j]):
                        is_critical = True
    if int(y_target[xyz_r]) not in collisions:
        is_critical = True
    return mask, visited, is_critical

def detect_critical_3d(y_target: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Detects critical components in a 3D segmentation using BFS on the mistakes mask.

    Parameters
    ----------
    y_target : np.ndarray
        Groundtruth segmentation.
    y_pred : np.ndarray
        Predicted segmentation.

    Returns
    -------
    int
        The number of critical components detected.
    """
    y_mistakes: np.ndarray = false_negative_mask(y_target, y_pred)
    y_target_minus_mistakes, _ = label(y_target * (1 - y_mistakes))
    n_criticals: int = 0
    foreground: Set[Tuple[int, int, int]] = get_foreground(y_mistakes)
    while foreground:
        xyz_r = sample(list(foreground), 1)[0]
        _, visited, is_critical = extract_component(y_target, y_mistakes, y_target_minus_mistakes, xyz_r)
        foreground = foreground.difference(visited)
        if is_critical:
            n_criticals += 1
    return n_criticals

def compute(label: np.ndarray, prediction: np.ndarray, **hyperparams: Any) -> Dict[str, float]:
    """
    Computes the numbers of positive and negative critical components.

    Positive critical components are computed as:
        detect_critical_3d(label, prediction)
    Negative critical components are computed as:
        detect_critical_3d(prediction, label)

    Parameters
    ----------
    label : np.ndarray
        Groundtruth segmentation.
    prediction : np.ndarray
        Predicted segmentation.
    hyperparams : Any
        Additional hyperparameters (currently unused).

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
            - "critical_components_positive": number of positive critical components.
            - "critical_components_negative": number of negative critical components.
    """
    pos_critical: int = detect_critical_3d(label, prediction)
    neg_critical: int = detect_critical_3d(prediction, label)

    return {
        "critical_components_positive": float(pos_critical),
        "critical_components_negative": float(neg_critical)
    }