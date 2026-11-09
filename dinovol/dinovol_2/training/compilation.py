"""runtime.compile settings shared by training methods."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CompileSettings:
    options: dict[str, Any]  # torch.compile keyword arguments except dynamic
    dynamic: dict[str, bool | None]  # per compiled module name

    def for_module(self, name: str) -> dict[str, Any]:
        return {**self.options, "dynamic": self.dynamic[name]}


def resolve_compile_settings(settings: Any, module_names: Sequence[str]) -> CompileSettings | None:
    """Validate runtime.compile (a boolean or a settings mapping); None when disabled.

    When enabled, also applies the process-wide Dynamo/Inductor configuration it implies.
    `dynamic` may map entries of `module_names` to settings; unlisted modules use null.
    """
    if isinstance(settings, bool):
        settings = {"enabled": settings}
    elif isinstance(settings, Mapping):
        settings = dict(settings)
    else:
        raise ValueError("compile must be a boolean or a settings mapping")
    unknown = set(settings) - {"enabled", "backend", "mode", "dynamic", "fullgraph", "optimize_ddp"}
    if unknown:
        raise ValueError(f"Unknown compile settings: {sorted(unknown)}")
    enabled = settings.pop("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("compile.enabled must be a boolean")
    if not enabled:
        return None
    # Dynamo's DDP optimizer splits compiled graphs at gradient-bucket boundaries. The split
    # graphs no longer honour the checkpoint save-attention policy: under DDP every student
    # attention forward ran again in backward (g256 crops, 3x H100: 5.75 -> 4.47 s/step off).
    # "python_reducer" instead traces DDP's reducer into a compiled-autograd backward.
    optimize_ddp = settings.pop("optimize_ddp", False)
    if not isinstance(optimize_ddp, bool) and optimize_ddp != "python_reducer":
        raise ValueError('compile.optimize_ddp must be a boolean or "python_reducer"')
    import torch._dynamo
    torch._dynamo.config.optimize_ddp = optimize_ddp
    if optimize_ddp == "python_reducer":
        torch._dynamo.config.compiled_autograd = True
    options = {"backend": "inductor", "mode": "default", "dynamic": None, "fullgraph": False, **settings}
    dynamic = options.pop("dynamic")
    if isinstance(dynamic, Mapping):
        unknown_modules = set(dynamic) - set(module_names)
        if unknown_modules:
            raise ValueError(f"Unknown compile.dynamic modules: {sorted(unknown_modules)}")
        dynamic_by_module = {name: dynamic.get(name) for name in module_names}
    else:
        dynamic_by_module = {name: dynamic for name in module_names}
    if any(value is not None and not isinstance(value, bool) for value in dynamic_by_module.values()):
        raise ValueError("compile.dynamic must be a boolean, null, or a mapping of module names to those")
    if not isinstance(options["fullgraph"], bool):
        raise ValueError("compile.fullgraph must be a boolean")
    if not isinstance(options["backend"], str) or not isinstance(options["mode"], str):
        raise ValueError("compile.backend and compile.mode must be strings")
    if options["backend"] == "inductor":
        # Inductor's mix-order reduction (torch 2.10) fuses the mask-token gradient with
        # LayerNormNd parameter gradients in the CNN backward and then returns it as [C]
        # instead of [1, 1, C] (seen with 8+ masked 128^3 crops). Disabling it cost no
        # measurable CNN throughput.
        import torch._inductor.config as inductor_config
        inductor_config.triton.mix_order_reduction = False
    return CompileSettings(options=options, dynamic=dynamic_by_module)
