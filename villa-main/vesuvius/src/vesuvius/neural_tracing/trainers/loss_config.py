from dataclasses import dataclass


@dataclass(frozen=True)
class TraceLossConfig:
    lambda_velocity_dir: float
    lambda_surface_attract: float
    lambda_trace_validity: float
    trace_validity_pos_weight: float
    lambda_velocity_smooth: float
    velocity_smooth_normalize: bool
    lambda_trace_integration: float
    trace_integration_steps: int
    trace_integration_step_size: float
    trace_integration_max_points: int
    trace_integration_min_weight: float
    trace_integration_detach_steps: bool
    surface_attract_huber_beta: float

    @classmethod
    def from_config(cls, config):
        loss_config = cls(
            lambda_velocity_dir=float(config.get('lambda_velocity_dir', 0.0)),
            lambda_surface_attract=float(config.get('lambda_surface_attract', 0.0)),
            lambda_trace_validity=float(config.get('lambda_trace_validity', 0.0)),
            trace_validity_pos_weight=float(config.get('trace_validity_pos_weight', 1.0)),
            lambda_velocity_smooth=float(config.get('lambda_velocity_smooth', 0.0)),
            velocity_smooth_normalize=bool(config.get('velocity_smooth_normalize', True)),
            lambda_trace_integration=float(config.get('lambda_trace_integration', 0.0)),
            trace_integration_steps=int(config.get('trace_integration_steps', 2)),
            trace_integration_step_size=float(config.get('trace_integration_step_size', 1.0)),
            trace_integration_max_points=int(config.get('trace_integration_max_points', 2048)),
            trace_integration_min_weight=float(config.get('trace_integration_min_weight', 0.5)),
            trace_integration_detach_steps=bool(config.get('trace_integration_detach_steps', False)),
            surface_attract_huber_beta=float(config.get('surface_attract_huber_beta', 5.0)),
        )
        if loss_config.trace_integration_steps < 0:
            raise ValueError(
                f"trace_integration_steps must be >= 0, got {loss_config.trace_integration_steps}"
            )
        if loss_config.trace_integration_step_size < 0.0:
            raise ValueError(
                "trace_integration_step_size must be >= 0, "
                f"got {loss_config.trace_integration_step_size}"
            )
        if loss_config.trace_integration_max_points < 0:
            raise ValueError(
                "trace_integration_max_points must be >= 0, "
                f"got {loss_config.trace_integration_max_points}"
            )
        return loss_config


@dataclass(frozen=True)
class CopyNeighborLossConfig:
    lambda_velocity_dir: float
    lambda_progress_phi: float
    lambda_progress_gradient: float
    lambda_stop: float
    lambda_surface_attract: float
    lambda_endpoint: float
    lambda_velocity_smooth: float
    velocity_smooth_normalize: bool
    stop_pos_weight: float
    surface_attract_huber_beta: float
    endpoint_steps: int
    endpoint_step_size: float
    endpoint_huber_beta: float
    endpoint_max_distance: float
    endpoint_detach_steps: bool

    @classmethod
    def from_config(cls, config):
        loss_config = cls(
            lambda_velocity_dir=float(config.get("lambda_copy_neighbor_velocity_dir", 1.0)),
            lambda_progress_phi=float(config.get("lambda_copy_neighbor_progress_phi", 0.25)),
            lambda_progress_gradient=float(config.get("lambda_copy_neighbor_progress_gradient", 0.25)),
            lambda_stop=float(config.get("lambda_copy_neighbor_stop", 0.25)),
            lambda_surface_attract=float(config.get("lambda_copy_neighbor_surface_attract", 0.1)),
            lambda_endpoint=float(config.get("lambda_copy_neighbor_endpoint", 0.25)),
            lambda_velocity_smooth=float(config.get("lambda_copy_neighbor_velocity_smooth", 0.0)),
            velocity_smooth_normalize=bool(config.get("copy_neighbor_velocity_smooth_normalize", True)),
            stop_pos_weight=float(config.get("copy_neighbor_stop_pos_weight", 1.0)),
            surface_attract_huber_beta=float(config.get("copy_neighbor_surface_attract_huber_beta", 5.0)),
            endpoint_steps=int(config.get("copy_neighbor_endpoint_steps", 8)),
            endpoint_step_size=float(config.get("copy_neighbor_endpoint_step_size", 1.0)),
            endpoint_huber_beta=float(config.get("copy_neighbor_endpoint_huber_beta", 2.0)),
            endpoint_max_distance=float(config.get("copy_neighbor_endpoint_max_distance", 32.0)),
            endpoint_detach_steps=bool(config.get("copy_neighbor_endpoint_detach_steps", False)),
        )
        if loss_config.endpoint_steps < 0:
            raise ValueError(f"copy_neighbor_endpoint_steps must be >= 0, got {loss_config.endpoint_steps}")
        if loss_config.endpoint_step_size < 0.0:
            raise ValueError(
                "copy_neighbor_endpoint_step_size must be >= 0, "
                f"got {loss_config.endpoint_step_size}"
            )
        if loss_config.endpoint_max_distance < 0.0:
            raise ValueError(
                "copy_neighbor_endpoint_max_distance must be >= 0, "
                f"got {loss_config.endpoint_max_distance}"
            )
        return loss_config
