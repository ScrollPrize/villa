SUPPORTED_MODEL_TYPES = (
    "timesformer",
    "resnet3d-50",
    "resnet3d-152-3d-decoder",
)

GPU_ONLY_STEPS = {"inference"}
CPU_SUPPORTED_STEPS = {"prepare", "reduce", "aggregate-profiling"}


def normalize_model_type(model_type: str) -> str:
    normalized = model_type.strip().lower()
    if normalized not in SUPPORTED_MODEL_TYPES:
        allowed = ", ".join(SUPPORTED_MODEL_TYPES)
        raise ValueError(f"MODEL_TYPE must be one of: {allowed}; got '{model_type}'")
    return normalized


def validate_image_role_for_step(step: str, image_role: str) -> None:
    normalized_role = image_role.strip().lower()
    if step in GPU_ONLY_STEPS and normalized_role == "cpu":
        raise RuntimeError(
            "STEP=inference requires the GPU image target; the current container is the CPU utility image."
        )
