from .base import StepOutput, TrainingMethod


def build_method(config, **runtime):
    name = config.get("method", {}).get("name", "dino")
    if name == "dino":
        from .dino import DinoMethod
        return DinoMethod(config, **runtime)
    if name == "jem":
        from .jem import JemMethod
        return JemMethod(config, **runtime)
    raise ValueError(f"Unknown training method: {name!r}")
