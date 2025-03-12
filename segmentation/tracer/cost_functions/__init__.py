"""Cost functions for volume cartographer tracer."""

# Import all cost functions from base implementations
from .base import (
    AnchorLoss,
    DistLoss,
    DistLoss2D,
    LinChkDistLoss,
    SpaceLineLossAcc,
    SpaceLossAcc,
    StraightLoss,
    StraightLoss2,
    StraightLoss2D,
    SurfaceLossD,
    ZCoordLoss,
    ZLocationLoss,
)

# Import all cost functions from autodiff implementations
from .autodiff import (
    AnchorLossAutoDiff,
    DistLossAutoDiff,
    DistLoss2DAutoDiff,
    LinChkDistLossAutoDiff,
    SpaceLineLossAccAutoDiff,
    SpaceLossAccAutoDiff,
    StraightLossAutoDiff,
    StraightLoss2AutoDiff,
    ZCoordLossAutoDiff,
    ZLocationLossAutoDiff,
)

# Import interpolators from core
from tracer.core.interpolation import (
    TrilinearInterpolator,
    TrilinearInterpolatorAutoDiff
)

__all__ = [
    # Base implementations
    "AnchorLoss",
    "DistLoss",
    "DistLoss2D",
    "LinChkDistLoss",
    "SpaceLineLossAcc",
    "SpaceLossAcc",
    "StraightLoss",
    "StraightLoss2",
    "StraightLoss2D",
    "SurfaceLossD",
    "ZCoordLoss",
    "ZLocationLoss",
    
    # AutoDiff implementations
    "AnchorLossAutoDiff",
    "DistLossAutoDiff",
    "DistLoss2DAutoDiff",
    "LinChkDistLossAutoDiff",
    "SpaceLineLossAccAutoDiff",
    "SpaceLossAccAutoDiff",
    "StraightLossAutoDiff",
    "StraightLoss2AutoDiff",
    "ZCoordLossAutoDiff",
    "ZLocationLossAutoDiff",
    
    # Utilities
    "TrilinearInterpolator",
    "TrilinearInterpolatorAutoDiff",
]