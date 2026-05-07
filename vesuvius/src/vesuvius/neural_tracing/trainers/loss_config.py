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
